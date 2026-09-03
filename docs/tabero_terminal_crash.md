# VS Code terminal closes with `free(): invalid pointer`

## Diagnosis on 2026-09-02

The failure was reproduced in an interactive `/bin/bash` without Python or VS Code:

| Input | Result |
| --- | --- |
| One-line `echo tabero-shell-probe` | Success, shell exits normally |
| `echo tabero-shell-probe` followed by a backslash and newline | `free(): invalid pointer`, SIGABRT |
| The same continued command sent as bracketed paste | Same SIGABRT |

The original user incident's precise timestamp is unknown. A local core from
15:23 on the same day, in this repository, identifies the failing native code:

```text
Executable: /usr/bin/bash
Signal: 6 (SIGABRT)
record_history_cmdline
  -> record_cmdline_core (libso/payload_x86_64_norm.c:677)
  -> free
  -> malloc_printerr("free(): invalid pointer")
  -> abort
```

`record_cmdline_core` belongs to `/usr/lib/libpayload.so` (also visible as
`/lib/libpayload.so` on this merged-/usr system). Its `loginName` local is the
fallback string `"-"` at an address in the library's read-only mapping. The failing
instruction passes that pointer to `free()`, although it is not heap storage.
At the failure, `multiLineOutPutFlag` is 0. This is a native memory-management bug
in the command-recording component.

Additional local evidence:

- `bash --version`: 5.1.16; dpkg package: `5.1-6ubuntu1.1`.
- `dpkg -V bash`: `??5?????? /bin/bash` (installed binary content differs from
  the package checksum).
- No dpkg package claims `/usr/lib/libpayload.so`.
- The library contains the build path
  `uhs/agent/go/clib/bashInjector/staticInjector/normInjector` and the path
  `/var/lib/UHS/historyOrder.log`. These identify a UHS-related command recorder;
  they do not establish the exact vendor or who installed it.
- Existing core report: `/var/crash/_usr_bin_bash.1009.crash`.

The library and system Bash were inspected read-only. No command-recording hooks,
system libraries, shell startup settings or security services were changed.

## Run the checks with one terminal command

From this repository, enter this **single line**:

```bash
bash scripts/run_tabero_checks.sh
```

This uses the existing Python environment and defaults to
`pi0_lora_tacfield_local_tactile_lora_smoke`. It executes preparation and batch
inspection from a script file, with CPU selection, native Python fault reporting
and separate timestamped logs. If preparation fails, batch inspection does not
start. It does not train a model. A different config can be supplied as its sole
argument.

The script invocation remains a normal shell command; it requires no changes to
the host's command recorder. The reproduced problem is interactive continuation
input. The underlying system component still needs repair by its maintainer.

## Scope of verification

- Both Python helpers passed `--help` with the installed environment.
- A CPU-only read of episode 0 passed: 201 Parquet rows; finite state/action
  `[201,7]` and tactile `[201,9,198,2]`; both videos decoded to 201 frames.
- The shell launcher passed syntax and help checks.
- Full dataset preparation, tokenizer download, actual transformed batches and
  training were not run during this diagnosis. The target audit/assets directories
  did not exist when inspected.

Provide the stack, library path and minimal continued-`echo` reproduction to the
server administrator or the command-recorder maintainer. They should fix the
invalid ownership/free of the fallback login-name string and verify interactive
multiline input. Changing JAX/PyTorch versions is not indicated by this crash.
