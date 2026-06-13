package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows"
)

const (
	serviceName = "DeltaMemGo"
	nssmPath    = `C:\Windows\System32\nssm.exe`
	nssmURL     = "https://nssm.cc/release/nssm-2.24.zip"
	defaultPort = "18080"
	defaultGRPC = "19090"
)

func defaultInstallDir() string {
	if isAdmin() {
		return `C:\Program Files\DeltaMemGo`
	}
	return filepath.Join(os.Getenv("APPDATA"), "mem-go")
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════╗")
	fmt.Println("║    δ-mem-go Installer v1.0                       ║")
	fmt.Println("║    Persistent Memory + Thought Synthesis Engine   ║")
	fmt.Println("╚══════════════════════════════════════════════════╝")
	fmt.Println()

	admin := isAdmin()
	if admin {
		fmt.Println("  Mode: SYSTEM-WIDE (Administrator)")
		fmt.Println("  → NSSM service, all users, auto-start on boot")
	} else {
		fmt.Println("  Mode: PER-USER (no admin required)")
		fmt.Println("  → Installed to %APPDATA%, runs at login via scheduled task")
	}
	fmt.Println()

	// 1. Install directory
	installDir := prompt("Install directory", defaultInstallDir())
	if err := os.MkdirAll(installDir, 0755); err != nil {
		fatal("Failed to create directory: %v", err)
	}

	// 2. Write binary
	exePath := filepath.Join(installDir, "delta-mem-go.exe")
	if len(agentBinary) == 0 {
		fatal("Embedded binary is empty — was the installer built correctly?")
	}
	if err := os.WriteFile(exePath, agentBinary, 0755); err != nil {
		fatal("Failed to write binary: %v", err)
	}
	fmt.Printf("  ✓ Installed delta-mem-go.exe (%d MB)\n", len(agentBinary)/1024/1024)

	// 3. Data directory
	dataDir := filepath.Join(installDir, "data")
	os.MkdirAll(dataDir, 0755)
	fmt.Printf("  ✓ Data directory: %s\n", dataDir)

	// 4. Model path
	fmt.Println()
	fmt.Println("  The ONNX embedding model (nomic-embed-text-v1.5) is required.")
	fmt.Println("  If you don't have it, download from: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5")
	modelPath := prompt("Path to nomic-embed-text-v1.5.onnx", filepath.Join(installDir, "models", "nomic-embed-text-v1.5.onnx"))
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Println("  ⚠ Model not found at that path. Service will start but embeddings will be disabled.")
		fmt.Println("    Place the model file there and restart the service.")
	} else {
		fmt.Println("  ✓ Model found")
	}

	// 5. ORT DLL
	ortPath := filepath.Join(filepath.Dir(modelPath), "onnxruntime.dll")
	if _, err := os.Stat(ortPath); os.IsNotExist(err) {
		fmt.Println("  ⚠ onnxruntime.dll not found next to model. Download from:")
		fmt.Println("    https://github.com/microsoft/onnxruntime/releases (match your Go library version)")
	} else {
		fmt.Println("  ✓ ONNX Runtime DLL found")
	}

	// 6. Port configuration
	httpPort := prompt("HTTP port", defaultPort)
	grpcPort := prompt("gRPC port", defaultGRPC)

	// 7. Owner (for initiation)
	owner := prompt("Owner name (your identifier)", os.Getenv("USERNAME"))

	// 8. Training data (optional)
	trainData := prompt("Training data file for initiation (optional, press Enter to skip)", "")

	// 9. Register service or scheduled task
	logFile := filepath.Join(installDir, "service.log")

	if admin {
		// System-wide: NSSM service
		if _, err := os.Stat(nssmPath); os.IsNotExist(err) {
			fmt.Println("  Downloading NSSM...")
			if err := downloadNSSM(); err != nil {
				fatal("Failed to download NSSM: %v", err)
			}
			fmt.Println("  ✓ NSSM installed")
		} else {
			fmt.Println("  ✓ NSSM already present")
		}
		nssm("install", serviceName, exePath)
		nssm("set", serviceName, "AppDirectory", installDir)
		nssm("set", serviceName, "AppParameters", fmt.Sprintf(`--model "%s" --port %s --grpc-port %s --data "%s"`, modelPath, httpPort, grpcPort, dataDir))
		nssm("set", serviceName, "DisplayName", "Delta-Mem-Go Thoughts Engine")
		nssm("set", serviceName, "Description", "Persistent memory + thought synthesis for AI agents.")
		nssm("set", serviceName, "Start", "SERVICE_AUTO_START")
		nssm("set", serviceName, "AppStdout", logFile)
		nssm("set", serviceName, "AppStderr", logFile)
		nssm("set", serviceName, "AppRotateFiles", "1")
		nssm("set", serviceName, "AppRotateBytes", "10485760")
		nssm("start", serviceName)
		fmt.Println("  ✓ NSSM service registered + started")
	} else {
		// Per-user: startup shortcut (no admin needed)
		startupDir := filepath.Join(os.Getenv("APPDATA"), `Microsoft\Windows\Start Menu\Programs\Startup`)
		batPath := filepath.Join(startupDir, "delta-mem-go.bat")
		batContent := fmt.Sprintf(`@echo off
start "" /B "%s" --model "%s" --port %s --grpc-port %s --data "%s" > "%s" 2>&1
`, exePath, modelPath, httpPort, grpcPort, dataDir, logFile)
		os.WriteFile(batPath, []byte(batContent), 0644)
		fmt.Println("  ✓ Startup shortcut created (runs at login)")
		// Also start now
		cmd := exec.Command(exePath, "--model", modelPath, "--port", httpPort, "--grpc-port", grpcPort, "--data", dataDir)
		cmd.Dir = installDir
		cmd.Start()
		fmt.Println("  ✓ Started (background)")
	}

	// 12. Summary
	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════")
	fmt.Println("  Installation complete!")
	fmt.Println()
	fmt.Printf("  Service:    %s\n", serviceName)
	fmt.Printf("  HTTP API:   http://localhost:%s\n", httpPort)
	fmt.Printf("  gRPC:       localhost:%s\n", grpcPort)
	fmt.Printf("  Data:       %s\n", dataDir)
	fmt.Printf("  Log:        %s\n", logFile)
	fmt.Printf("  Owner:      %s\n", owner)
	fmt.Println()
	fmt.Println("  Quick test:")
	fmt.Printf("    curl http://localhost:%s/health\n", httpPort)
	fmt.Println()
	fmt.Println("  Commands:")
	fmt.Println("    nssm status DeltaMemGo")
	fmt.Println("    nssm restart DeltaMemGo")
	fmt.Println("    nssm stop DeltaMemGo")
	fmt.Println()
	if trainData != "" {
		fmt.Println("  To initiate with your data, run:")
		fmt.Printf("    delta-mem-go.exe --initiate --owner %s --training-data \"%s\"\n", owner, trainData)
	} else {
		fmt.Println("  To initiate (first-time training on your domain data):")
		fmt.Printf("    delta-mem-go.exe --initiate --owner %s --training-data <your-file.txt>\n", owner)
	}
	fmt.Println("══════════════════════════════════════════════════")
}

func isAdmin() bool {
	var sid *windows.SID
	err := windows.AllocateAndInitializeSid(
		&windows.SECURITY_NT_AUTHORITY, 2,
		windows.SECURITY_BUILTIN_DOMAIN_RID, windows.DOMAIN_ALIAS_RID_ADMINS,
		0, 0, 0, 0, 0, 0, &sid)
	if err != nil { return false }
	defer windows.FreeSid(sid)
	member, err := windows.Token(0).IsMember(sid)
	return err == nil && member
}

func prompt(label, defaultVal string) string {
	reader := bufio.NewReader(os.Stdin)
	if defaultVal != "" {
		fmt.Printf("  %s [%s]: ", label, defaultVal)
	} else {
		fmt.Printf("  %s: ", label)
	}
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)
	if input == "" { return defaultVal }
	return input
}

func nssm(args ...string) {
	cmd := exec.Command(nssmPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("  ⚠ nssm %s: %v\n", strings.Join(args, " "), err)
	}
}

func downloadNSSM() error {
	resp, err := http.Get(nssmURL)
	if err != nil { return err }
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil { return err }
	zr, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil { return err }
	for _, f := range zr.File {
		if strings.HasSuffix(f.Name, "win64/nssm.exe") {
			rc, _ := f.Open()
			defer rc.Close()
			data, _ := io.ReadAll(rc)
			return os.WriteFile(nssmPath, data, 0755)
		}
	}
	return fmt.Errorf("nssm.exe not found in archive")
}

func fatal(format string, args ...interface{}) {
	fmt.Printf("\n  ✗ ERROR: "+format+"\n", args...)
	os.Exit(1)
}
