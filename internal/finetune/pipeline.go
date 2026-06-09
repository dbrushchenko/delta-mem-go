package finetune

import (
	"context"
	"fmt"
	"os/exec"
)

type Pipeline struct {
	UnslothPath string
}

func NewPipeline(unslothPath string) *Pipeline {
	return &Pipeline{UnslothPath: unslothPath}
}

func (p *Pipeline) Run(ctx context.Context, datasetPath, outputModelPath string, epochs int) error {
	cmd := exec.CommandContext(ctx, "python", p.UnslothPath,
		"--model", "gemma-4-e4b-it-q4",
		"--data", datasetPath,
		"--output", outputModelPath,
		"--epochs", fmt.Sprintf("%d", epochs),
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("fine-tuning failed: %w\noutput: %s", err, output)
	}
	return nil
}
