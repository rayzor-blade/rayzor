//! Host-side system operations; child output never uses the LSP transport stream.

use super::*;

impl MacroInterpreter {
    pub(super) fn host_process_output(
        &self,
        operation: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<std::process::Output, MacroError> {
        let error = |message: &str| MacroError::TypeError {
            message: format!("{operation}: {message}"),
            location,
        };
        let command = args
            .first()
            .and_then(MacroValue::as_string)
            .ok_or_else(|| error("expected a command string"))?;
        let mut process = match args.get(1) {
            Some(MacroValue::Array(arguments)) => {
                let mut process = std::process::Command::new(command);
                for argument in arguments.iter() {
                    process.arg(
                        argument
                            .as_string()
                            .ok_or_else(|| error("arguments must be strings"))?,
                    );
                }
                process
            }
            None | Some(MacroValue::Null) => {
                #[cfg(windows)]
                let mut process = {
                    let mut process = std::process::Command::new("cmd");
                    process.arg("/C");
                    process
                };
                #[cfg(not(windows))]
                let mut process = {
                    let mut process = std::process::Command::new("/bin/sh");
                    process.arg("-c");
                    process
                };
                process.arg(command);
                process
            }
            _ => return Err(error("expected an optional array of arguments")),
        };
        process.output().map_err(|error| MacroError::RuntimeError {
            message: format!("{operation}('{command}') failed: {error}"),
            location,
        })
    }

    pub(super) fn system_call(
        &mut self,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        Ok(Some(match method {
            "command" => {
                let output = self.host_process_output("Sys.command", args, location)?;
                use std::io::Write;
                let mut stderr = std::io::stderr().lock();
                let _ = stderr.write_all(&output.stdout);
                let _ = stderr.write_all(&output.stderr);
                MacroValue::Int(output.status.code().unwrap_or(-1) as i64)
            }
            "print" | "println" => {
                self.trace_output.push(
                    args.first()
                        .map(MacroValue::to_display_string)
                        .unwrap_or_else(|| "null".into()),
                );
                MacroValue::Null
            }
            "getEnv" => {
                let name = args
                    .first()
                    .and_then(MacroValue::as_string)
                    .ok_or_else(|| MacroError::TypeError {
                        message: "Sys.getEnv expects a string".into(),
                        location,
                    })?;
                std::env::var_os(name)
                    .map(|value| MacroValue::from_str(&value.to_string_lossy()))
                    .unwrap_or(MacroValue::Null)
            }
            "getCwd" => MacroValue::from_string(
                std::env::current_dir()
                    .map_err(|error| MacroError::RuntimeError {
                        message: format!("Sys.getCwd failed: {error}"),
                        location,
                    })?
                    .to_string_lossy()
                    .into_owned(),
            ),
            "systemName" => MacroValue::from_str(if cfg!(target_os = "macos") {
                "Mac"
            } else if cfg!(target_os = "linux") {
                "Linux"
            } else if cfg!(windows) {
                "Windows"
            } else {
                "Unknown"
            }),
            "programPath" | "executablePath" => MacroValue::from_string(
                std::env::current_exe()
                    .map_err(|error| MacroError::RuntimeError {
                        message: format!("Sys.{method} failed: {error}"),
                        location,
                    })?
                    .to_string_lossy()
                    .into_owned(),
            ),
            "time" => MacroValue::Float(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|error| MacroError::RuntimeError {
                        message: format!("Sys.time failed: {error}"),
                        location,
                    })?
                    .as_secs_f64(),
            ),
            _ => return Ok(None),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(expression: &str) -> Result<MacroValue, MacroError> {
        let tokens = parser::lexer::Lexer::new(expression).tokenize().unwrap();
        let mut stream = parser::token_stream::TokenStream::new(&tokens, expression);
        let expr = parser::rd::RdParser::new(&mut stream, expression, "<system-test>")
            .parse_expression()
            .unwrap();
        MacroInterpreter::new(MacroRegistry::new()).eval_expr(&expr)
    }

    #[test]
    fn system_command_without_arguments_uses_the_host_shell() {
        assert_eq!(
            evaluate("Sys.command(\"exit 9\")").unwrap(),
            MacroValue::Int(9)
        );
    }

    #[test]
    #[cfg(unix)]
    fn system_command_preserves_argument_boundaries_and_exit_status() {
        let argument = "a b;$(printf expanded)";
        let script = "if [ \"$1\" = 'a b;$(printf expanded)' ]; then exit 7; else exit 8; fi";
        let expression = format!(
            "Sys.command(\"/bin/sh\", [\"-c\", {}, \"macro\", {}])",
            serde_json::to_string(script).unwrap(),
            serde_json::to_string(argument).unwrap()
        );
        assert_eq!(evaluate(&expression).unwrap(), MacroValue::Int(7));
    }

    #[test]
    fn promoted_system_calls_fall_back_to_host_execution() {
        let file = parser::parse_haxe_file(
            "Host.hx",
            "class Host { public static macro function check():Int return Sys.command('exit 7'); }",
            false,
        )
        .unwrap();
        let mut registry = MacroRegistry::new();
        registry.scan_and_register(&file, &file.filename).unwrap();
        let definition = registry.get_macro("Host.check").unwrap().clone();
        let mut interpreter = MacroInterpreter::new(registry);
        interpreter.vm = Some(MacroVm::new());
        interpreter.scheduler = Some(MorselScheduler {
            call_counts: BTreeMap::new(),
            threshold: 1,
            classes_compiled: false,
        });
        for _ in 0..3 {
            assert_eq!(
                interpreter
                    .call_macro_def(&definition, vec![], SourceLocation::unknown())
                    .unwrap(),
                MacroValue::Int(7)
            );
        }
        assert!(interpreter.registry.get_compiled("Host.check").is_some());
    }

    #[test]
    fn absent_environment_variables_remain_null() {
        assert_eq!(
            evaluate("Sys.getEnv(\"__RAYZOR_MACRO_MISSING_ENV__\")").unwrap(),
            MacroValue::Null
        );
    }
}
