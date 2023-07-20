## CodeGen Setup using SHARK-server

### Setup Server
- clone SHARK and setup the venv
- host the server using `python apps/stable_diffusion/web/index.py --api --server_port=<PORT>`
- default server address is `http://0.0.0.0:8080`

### Setup Client
1. fauxpilot-vscode (VSCode Extension):
- Code for the extension can be found [here](https://github.com/Venthe/vscode-fauxpilot)
- PreReq: VSCode extension (will need [`nodejs` and `npm`](https://nodejs.org/en/download) to compile and run the extension)
- Compile and Run the extension on VSCode (press F5 on VSCode), this opens a new VSCode window with the extension running
- Open VSCode settings, search for fauxpilot in settings and modify `server : http://<IP>:<PORT>`, `Model : codegen` , `Max Lines : 30`

2. Others (REST API curl, OpenAI Python bindings) as shown [here](https://github.com/fauxpilot/fauxpilot/blob/main/documentation/client.md)
- using Github Copilot VSCode extension with SHARK-server needs more work to be functional.