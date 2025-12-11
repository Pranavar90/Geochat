# Setting up Ollama for GeoChat

Since you are on Windows, follow these steps to get your LLM running:

## 1. Download & Install
1.  Go to **[ollama.com/download](https://ollama.com/download)**.
2.  Click "Download for Windows".
3.  Run the installer (`OllamaSetup.exe`).

## 2. Pull the Model
We are using the **3.8B Mini** model for speed and efficiency.
Open your **PowerShell** or **Command Prompt** and run:

```powershell
ollama pull phi3:mini
```

## 3. Verify it's Running
Run this command to check if the server is active:
```powershell
ollama list
```
You should see `phi3:mini` in the list.

## 4. Test (Optional)
You can chat with it directly to test:
```powershell
ollama run phi3:mini "Hello, are you ready?"
```
(Type `/bye` to exit).

**That's it!** Once checking is done, my scripts (`rag_engine.py`, `geochat.py`) will automatically connect to it.
