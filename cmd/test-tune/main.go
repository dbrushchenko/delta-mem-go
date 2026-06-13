package main

import (
	"fmt"
	"os"
	"time"

	"github.com/dbrushchenko/delta-mem-go/internal/deltamem"
	"github.com/dbrushchenko/delta-mem-go/internal/embeddings"
	"github.com/dbrushchenko/delta-mem-go/internal/ibnn"
	"github.com/dbrushchenko/delta-mem-go/internal/thoughts"
	"github.com/dbrushchenko/delta-mem-go/internal/turbovec"
)

func main() {
	emb, _ := embeddings.Get(`C:\Users\dabrush\mem-go\models\nomic-embed-text-v1.5.onnx`, `C:\Users\dabrush\mem-go\models\onnxruntime.dll`)
	defer emb.Close()
	thoughts.SetEmbedder(emb)
	data, _ := os.ReadFile(`C:\Users\dabrush\mem-go\training_3.txt`)

	configs := []struct{ epochs int; lr float32 }{
		{3, 0.01}, {5, 0.01}, {10, 0.01},
		{3, 0.05}, {5, 0.05}, {10, 0.05},
		{3, 0.1}, {5, 0.1},
	}

	fmt.Printf("%-12s %-8s %-10s %-10s %-10s\n", "Config", "LR", "Norm", "AvgConf", "Time")
	fmt.Println("-------------------------------------------------------")

	for _, c := range configs {
		os.RemoveAll("./data/tune_test")
		os.MkdirAll("./data/tune_test", 0755)
		deltaOM := deltamem.NewOwnerManager(deltamem.Config{R: 64, HiddenDim: 768, NormCap: 10.0}, "./data/tune_test")
		ibnnOM := ibnn.NewOwnerManager(ibnn.Config{InputDim: 768, HiddenDim: 768}, "./data/tune_test")
		turboOM := turbovec.NewOwnerManager(768)
		engine := thoughts.New(deltaOM, ibnnOM, turboOM, nil)

		start := time.Now()
		r, err := engine.Initiate(string(data), "tune", thoughts.InitConfig{Epochs: c.epochs, LearningRate: c.lr, ChunkSize: 200})
		if err != nil {
			fmt.Printf("epochs=%d lr=%.2f FAIL: %v\n", c.epochs, c.lr, err)
			continue
		}
		fmt.Printf("epochs=%-4d  lr=%.2f  norm=%-8.4f conf=%-8.4f %v\n",
			c.epochs, c.lr, r.FinalNorm, r.AvgConf, time.Since(start).Round(time.Second))
	}
}
