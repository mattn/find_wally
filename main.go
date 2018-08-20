package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func getNormalizedGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))
	output = op.ExpandDims(s,
		op.Cast(s, decode, tf.Uint8),
		op.Const(s.SubScope("axis"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func getNormalizedImageTensor(buf *bytes.Buffer) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(buf.String())
	if err != nil {
		return nil, fmt.Errorf("cannot make tensor from bytes: %v", err)
	}

	graph, input, output, err := getNormalizedGraph()
	if err != nil {
		return nil, fmt.Errorf("cannot make normalize graph: %v", err)
	}
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatalf("could not run inference: %v", err)
		return nil, fmt.Errorf("cannot make session from graph: %v", err)
	}
	result, err := session.Run(
		map[tf.Output]*tf.Tensor{
			input: tensor,
		},
		[]tf.Output{
			output,
		},
		nil)
	if err != nil {
		return nil, fmt.Errorf("cannot run normalize graph: %v", err)
	}
	return result[0], nil
}

type circle struct {
	b image.Rectangle
	p image.Point
	r int
}

func (c *circle) ColorModel() color.Model {
	return color.AlphaModel
}

func (c *circle) Bounds() image.Rectangle {
	return c.b
}

func (c *circle) At(x, y int) color.Color {
	xx, yy, rr := float64(x-c.p.X)+0.5, float64(y-c.p.Y)+0.5, float64(c.r)
	if xx*xx+yy*yy < rr*rr {
		return color.Alpha{255}
	}
	return color.Alpha{100}
}

func main() {
	var modelfile string
	var output string

	flag.StringVar(&output, "output", "output.jpg", "output file")
	flag.StringVar(&modelfile, "model", "", "model file")
	flag.Parse()
	if flag.NArg() != 1 || modelfile == "" {
		flag.Usage()
		os.Exit(2)
	}

	var buf bytes.Buffer
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	io.Copy(&buf, f)

	input, err := getNormalizedImageTensor(&buf)
	if err != nil {
		log.Fatal(err)
	}

	model, err := ioutil.ReadFile(modelfile)
	if err != nil {
		log.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	tensor_boxes := graph.Operation("detection_boxes").Output(0)
	tensor_scores := graph.Operation("detection_scores").Output(0)
	tensor_num_detections := graph.Operation("num_detections").Output(0)

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	result, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("image_tensor").Output(0): input,
		},
		[]tf.Output{
			tensor_boxes,
			tensor_scores,
			tensor_num_detections,
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}
	num_detections := int(result[2].Value().([]float32)[0])
	boxes := result[0].Value().([][][]float32)[0][:num_detections]
	scores := result[1].Value().([][]float32)[0][:num_detections]

	img, err := jpeg.Decode(&buf)
	if err != nil {
		log.Fatal(err)
	}
	canvas := image.NewRGBA(img.Bounds())
	bounds := canvas.Bounds()
	for i, box := range boxes {
		if scores[i] < 0.9 {
			continue
		}

		ymin, xmin, ymax, xmax :=
			int(float32(bounds.Dy())*box[0]),
			int(float32(bounds.Dx())*box[1]),
			int(float32(bounds.Dy())*box[2]),
			int(float32(bounds.Dx())*box[3])
		c := &circle{
			bounds,
			image.Point{(xmin + xmax) / 2, (ymin + ymax) / 2},
			30,
		}
		draw.DrawMask(canvas, bounds, img, image.ZP, c, image.ZP, draw.Over)
	}
	out, err := os.Create(output)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()

	err = jpeg.Encode(out, canvas, nil)
	if err != nil {
		log.Fatal(err)
	}
}
