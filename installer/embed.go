//go:build !dev

package main

import _ "embed"

//go:embed delta-mem-go.exe
var agentBinary []byte

// hookBinary is optionally embedded (built separately)
var hookBinary []byte
