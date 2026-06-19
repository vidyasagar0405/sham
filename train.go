package main

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
)

// TYPES

type Class string

const (
	Spam = "Spam"
	Ham  = "Ham"
)

type Record struct {
	Class    Class
	Contents string
}

type dataset []Record

type Vocab map[string]bool

// Model holds pre-computed Log10 probabilities of Seen words
type Model map[string]float64

type ClassifierData struct {
	SpamLogs    Model
	HamLogs     Model
	DefaultSpam float64
	DefaultHam  float64
	PSpamRecord float64
	PHamRecord  float64
}

type Bow map[string]float64

func NewBow() Bow {
	return make(map[string]float64)
}

func ParseFile(datasetFile string) dataset {
	file, err := os.Open(datasetFile)
	if err != nil {
		log.Fatal("Error opening file:", err)
	}
	defer file.Close()

	r := csv.NewReader(file)
	records, err := r.ReadAll()
	if err != nil {
		log.Fatal("Error reading CSV:", err)
	}

	var data dataset
	for i, row := range records {
		if i == 0 {
			continue
		}
		data = append(data, Record{
			Class:    Class(row[0]),
			Contents: row[1],
		})
	}
	return data
}

// SplitDataset takes the full dataset and a ratio (e.g., 0.8 for 80% train / 20% test)
// and returns two separate datasets.
func SplitDataset(data dataset, trainRatio float64) (dataset, dataset) {
	// Create a copy of the dataset so we don't accidentally mutate the original
	shuffled := make(dataset, len(data))
	copy(shuffled, data)

	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	// Calculate the index where we make the cut
	splitIndex := int(float64(len(shuffled)) * trainRatio)

	// Slice the array into two distinct chunks
	trainSet := shuffled[:splitIndex]
	testSet := shuffled[splitIndex:]

	return trainSet, testSet
}

func makeBOWs(records dataset) (Bow, Bow, float64, float64) {
	spamBOW := NewBow()
	hamBOW := NewBow()
	spamCount := 0.0
	hamCount := 0.0

	for _, record := range records {
		words := strings.Fields(record.Contents)

		if record.Class == Spam {
			spamCount++
			for _, w := range words {
				spamBOW[w]++
			}
		}
		if record.Class == Ham {
			hamCount++
			for _, w := range words {
				hamBOW[w]++
			}
		}
	}
	return spamBOW, hamBOW, spamCount, hamCount
}

func totalWords(b Bow) float64 {
	count := 0.0
	for _, v := range b {
		count += v
	}
	return count
}

func buildVocab(spamBOW Bow, hamBOW Bow) Vocab {
	vocab := make(Vocab)
	for word := range spamBOW {
		vocab[word] = true
	}
	for word := range hamBOW {
		vocab[word] = true
	}
	return vocab
}

func getVocabSize(spamBOW Bow, hamBOW Bow) float64 {
	return float64(len(buildVocab(spamBOW, hamBOW)))
}

func LogPWordClass(word string, bow Bow, totalBOWWords float64, lenVocabulary float64) float64 {
	return math.Log10(bow[word]+1.0) / (totalBOWWords + lenVocabulary)
}

func WhatIsIt(pWordSpam float64, pWordHam float64) Class {
	if pWordHam > pWordSpam {
		return Ham
	} else {
		return Spam
	}
}

func TrainModel(spamBOW, hamBOW Bow, totalSpam, totalHam, vocabSize float64) (Model, Model, float64, float64) {
	spamLogProbs := make(Model)
	hamLogProbs := make(Model)

	// Unseen Word default probability (Count = 0)
	defaultSpamLog := math.Log10(1.0 / (totalSpam + vocabSize))
	defaultHamLog := math.Log10(1.0 / (totalHam + vocabSize))

	// Rebuild the vocabulary list to iterate through every known word
	vocab := buildVocab(spamBOW, hamBOW)
	// Pre-compute the math for every word
	for w := range vocab {
		spamLogProbs[w] = math.Log10((spamBOW[w] + 1.0) / (totalSpam + vocabSize))
		hamLogProbs[w] = math.Log10((hamBOW[w] + 1.0) / (totalHam + vocabSize))
	}

	return spamLogProbs, hamLogProbs, defaultSpamLog, defaultHamLog
}

func pEmail(r Record, spamLogs, hamLogs Model, defaultSpamLog, defaultHamLog, logPriorSpam, logPriorHam float64) (float64, float64) {
	words := strings.Fields(r.Contents)

	pEmailSpam := logPriorSpam
	pEmailHam := logPriorHam

	for _, w := range words {
		// Check if the word exists in our pre-computed Spam model
		if val, exists := spamLogs[w]; exists {
			pEmailSpam += val
		} else {
			pEmailSpam += defaultSpamLog // Fast fallback for unseen words
		}

		// Check the Ham model
		if val, exists := hamLogs[w]; exists {
			pEmailHam += val
		} else {
			pEmailHam += defaultHamLog
		}
	}

	return pEmailSpam, pEmailHam
}

func CalcStats(testRecords dataset, spamLogs, hamLogs Model, defaultSpam, defaultHam, pSpamRecord, pHamRecord float64) {
	// Set up the Confusion Matrix counters
	tp := 0.0 // True Positive: Model guessed Spam, actually Spam
	tn := 0.0 // True Negative: Model guessed Ham, actually Ham
	fp := 0.0 // False Positive: Model guessed Spam, actually Ham
	fn := 0.0 // False Negative: Model guessed Ham, actually Spam

	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, e := range testRecords {
		wg.Add(1)

		go func(e Record) {
			defer wg.Done()
			pemailSpam, pemailHam := pEmail(e, spamLogs, hamLogs, defaultSpam, defaultHam, pSpamRecord, pHamRecord)

			trueClass := e.Class
			predictClass := WhatIsIt(pemailSpam, pemailHam)

			// Populate the matrix
			mu.Lock()
			if predictClass == Spam && trueClass == Spam {
				tp++
			} else if predictClass == Ham && trueClass == Ham {
				tn++
			} else if predictClass == Spam && trueClass == Ham {
				fp++
			} else if predictClass == Ham && trueClass == Spam {
				fn++
			}
			mu.Unlock()
		}(e)
	}

	wg.Wait()

	// Calculate the standard ML metrics
	// Accuracy: (TP + TN) / Total
	accuracy := (tp + tn) / float64(len(testRecords))

	// Precision: TP / (TP + FP)
	precision := 0.0
	if (tp + fp) > 0 {
		precision = tp / (tp + fp)
	}

	// Recall: TP / (TP + FN)
	recall := 0.0
	if (tp + fn) > 0 {
		recall = tp / (tp + fn)
	}

	// F1 Score: Harmonic mean of Precision and Recall
	f1Score := 0.0
	if (precision + recall) > 0 {
		f1Score = 2 * ((precision * recall) / (precision + recall))
	}

	fmt.Println("=== Model Evaluation Dashboard ===")
	fmt.Printf("Total Tested : %v\n\n", len(testRecords))

	fmt.Printf("True Positives (Spam caught) : %v\n", tp)
	fmt.Printf("True Negatives (Ham kept)    : %v\n", tn)
	fmt.Printf("False Positives (Ham trashed): %v\n", fp)
	fmt.Printf("False Negatives (Spam leaked): %v\n\n", fn)

	fmt.Printf("Accuracy  : %.4f\n", accuracy)
	fmt.Printf("Precision : %.4f\n", precision)
	fmt.Printf("Recall    : %.4f\n", recall)
	fmt.Printf("F1 Score  : %.4f\n", f1Score)
	fmt.Println("==================================")
}

func SaveModel(filename string, data ClassifierData) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Could not create model file:", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(data)
	if err != nil {
		log.Fatal("Could not encode model:", err)
	}
}

func LoadModel(filename string) ClassifierData {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal("Could not open model file:", err)
	}
	defer file.Close()

	var data ClassifierData

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&data)
	if err != nil {
		log.Fatal("Could not decode model:", err)
	}

	return data
}

func main() {
	datasetFile := "./dataset/spam_Emails_data.csv"

	records := ParseFile(datasetFile)
	totalRecords := float64(len(records))

	trainRecords, testRecords := SplitDataset(records, 0.80)
	spamBOW, hamBOW, spamRecordCount, hamRecordCount := makeBOWs(trainRecords)

	totalSpamWords := totalWords(spamBOW)
	totalHamWords := totalWords(hamBOW)

	vocabSize := getVocabSize(spamBOW, hamBOW)
	pHamRecord := hamRecordCount / totalRecords   // Prior Probability
	pSpamRecord := spamRecordCount / totalRecords // Prior Probability

	spamLogs, hamLogs, defaultSpam, defaultHam := TrainModel(spamBOW, hamBOW, totalSpamWords, totalHamWords, vocabSize)

	CalcStats(testRecords, spamLogs, hamLogs, defaultSpam, defaultHam, pSpamRecord, pHamRecord)

	classifierData := ClassifierData{
		SpamLogs:    spamLogs,
		HamLogs:     hamLogs,
		DefaultSpam: defaultSpam,
		DefaultHam:  defaultHam,
		PSpamRecord: pSpamRecord,
		PHamRecord:  pHamRecord,
	}
	SaveModel("./shamModel.gob", classifierData)
}
