
### CLI

The package comes with a few CLI commands to help you manage model files.

```bash
$ infs --help
infs <command>

Commands:
  infs list [configPath]     List stored models               [aliases: ls, dir]
  infs show <modelName>      Print details of a model   [aliases: info, details]
  infs prepare <configPath>  Prepare models defined in configuration
                                                       [aliases: prep, download]
  infs remove <pattern>      Delete models matching the pattern
                                                              [aliases: rm, del]
```

#### List

Print currently stored models.

```bash
$ infs list
Models cache path:  /home/user/.cache/inference-server/models
Total cache size:   392 GB

156 files:
└── huggingface.co (342 GB)
    ├── Combatti (2.02 GB)
    │   └── llama3.2-3B-FunctionCalling-main (2.02 GB)
    ├── Comfy-Org (6.53 GB)
    │   └── stable-diffusion-3.5-fp8-main (6.53 GB)
    │       └── text_encoders (6.53 GB)
    ├── HuggingFaceTB (1.86 GB)
    │   ├── SmolLM2-1.7B-Instruct-main (1.72 GB)
    │   │   └── onnx (1.71 GB)
    │   └── smollm-135M-instruct-v0.2-Q8_0-GGUF-main (145 MB)
```

Positional arguments:
- `configPath`: Path to the configuration file. If not specified the command will look for `'infs.config.js', 'infs.config.mjs', 'infs.config.json', 'package.json'` in the current working directory. `list` will only print models that are defined in the configuration file. If no configuration file is found, all models in the cache will be shown.

Flags:
- `--all` `-a`: Show all models in cache, independently of the configuration file.
- `--json` `-j`: Output in JSON format.
- `--files` `-f`: Only directories will be shown by default. Use this flag to show files as well.
- `--list` `-l`: Output as flat list instead of tree for easier parsing and copying.

#### Remove

Delete one or more models from cache by their path pattern.

```bash
$ infs remove huggingface.co/Combatti/llama3.2-3B-FunctionCalling-main
└── llama3.2-3B-FunctionCalling-main (2.02 GB)
    └── unsloth.Q4_K_M.gguf (2.02 GB)

This will remove one file freeing 2.02 GB total.
Delete huggingface.co/Combatti/llama3.2-3B-FunctionCalling-main from disk? (y/N): y
Deleting llama3.2-3B-FunctionCalling-main ...
Done
```

Note that this commands takes a glob pattern. For example, to delete only certain quants, or all of a hub organization's models.

Positional arguments:
- `pattern`: Path pattern to match models to delete. Use `infs list` to see the available models.

Flags:
- `--yes` `-y`: Skip confirmation prompt.

#### Prepare

Run preparation tasks for all models defined in the configuration file. This command will download and validate the model files.

```bash
$ infs prepare tests/testmodels.config.js
```

Positional arguments:
- `configPath`: Path to the configuration file. If not specified the command will look for `'infs.config.js', 'infs.config.mjs', 'infs.config.json', 'package.json'` in the current working directory.

Flags:
- `--concurrency` `-c`: Number of concurrent preparation tasks. Default is 1.