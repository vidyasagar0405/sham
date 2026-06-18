package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// We redefine the exact same struct we used to save the model
type Model map[string]float64

type ClassifierData struct {
	SpamLogs     Model
	HamLogs      Model
	DefaultSpam  float64
	DefaultHam   float64
	LogPriorSpam float64
	LogPriorHam  float64
}

// LoadModel opens the binary file and decodes it back into our struct
func LoadModel(filename string) ClassifierData {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Could not open model file. Did you run the training script first? Error: %v", err)
	}
	defer file.Close()

	var data ClassifierData
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		log.Fatalf("Could not decode model: %v", err)
	}

	return data
}

// We update Predict to accept a "label" so we know which file we are looking at in the output
func Predict(label string, email string, data ClassifierData) {
	words := strings.Fields(strings.ToLower(email))

	pSpam := data.LogPriorSpam
	pHam := data.LogPriorHam

	for _, w := range words {
		if val, exists := data.SpamLogs[w]; exists {
			pSpam += val
		} else {
			pSpam += data.DefaultSpam
		}

		if val, exists := data.HamLogs[w]; exists {
			pHam += val
		} else {
			pHam += data.DefaultHam
		}
	}

	// Output now includes the filename
	if pSpam > pHam {
		fmt.Printf("[\033[31mSPAM\033[0m] %s | Score: %.2f (Ham: %.2f)\n", label, pSpam, pHam)
	} else {
		fmt.Printf("[\033[32mHAM\033[0m]  %s | Score: %.2f (Spam: %.2f)\n", label, pHam, pSpam)
	}
}

func main() {
	modelPath := flag.String("model", "./shamModel.gob", "Path to the trained model binary (.gob file)")

	flag.Parse()

	args := flag.Args()

	if len(args) < 1 {
		fmt.Println("Usage: ./predict [--model path/to/model.gob] <file1.txt> <file2.txt> ...")
		os.Exit(1)
	}

	modelData := LoadModel(*modelPath)

	for _, filename := range args {
		content, err := os.ReadFile(filename)
		if err != nil {
			fmt.Printf("[\033[33mERROR\033[0m] Skipping %s: %v\n", filename, err)
			continue
		}

		Predict(filename, string(content), modelData)
	}
}
