package turbogo

var (
	centroids2Bit = []float32{-0.798, -0.319, 0.319, 0.798}
	centroids4Bit = []float32{
		-0.95, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25,
		0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
	}
)

func Quantize(x float32, bitWidth int) byte {
	if bitWidth == 2 {
		for i := 0; i < 4; i++ {
			if x < centroids2Bit[i] {
				return byte(i)
			}
		}
		return 3
	}
	for i := 0; i < 16; i++ {
		if x < centroids4Bit[i] {
			return byte(i)
		}
	}
	return 15
}

func Dequantize(code byte, bitWidth int) float32 {
	if bitWidth == 2 {
		return centroids2Bit[code]
	}
	return centroids4Bit[code]
}
