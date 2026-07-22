use std::path::PathBuf;
use anyhow::{Context, Result, bail};
use crate::cli::{Cli, LlmProvider};

pub struct Config {
    pub ontology: PathBuf,
    pub shapes: PathBuf,
    pub corpus: PathBuf,
    pub out_kg: PathBuf,
    pub out_report: PathBuf,
    pub max_tool_calls: u32,
    pub rudof_bin: PathBuf,
    pub llm: LlmConfig,
}

pub struct LlmConfig {
    #[allow(dead_code)]
    pub provider: LlmProvider,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Config {
    pub fn from_cli(cli: Cli) -> Result<Self> {
        let llm = resolve_llm(cli.model)?;
        for p in [&cli.ontology, &cli.shapes, &cli.corpus] {
            if !p.exists() {
                bail!("input path does not exist: {}", p.display());
            }
        }
        if !cli.rudof_bin.exists() {
            bail!(
                "rudof binary not found at {} — pass --rudof-bin or set RUDOF_BIN",
                cli.rudof_bin.display()
            );
        }
        Ok(Self {
            ontology: cli.ontology,
            shapes: cli.shapes,
            corpus: cli.corpus,
            out_kg: cli.out_kg,
            out_report: cli.out_report,
            max_tool_calls: cli.max_tool_calls,
            rudof_bin: cli.rudof_bin,
            llm,
        })
    }
}

fn resolve_llm(provider: LlmProvider) -> Result<LlmConfig> {
    let (key_var, base_var, model_var, default_base, default_model) = match provider {
        LlmProvider::Qwen => (
            "QWEN_API_KEY",
            "QWEN_BASE_URL",
            "QWEN_MODEL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen-plus",
        ),
        LlmProvider::Deepseek => (
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL",
            "https://api.deepseek.com/v1",
            "deepseek-chat",
        ),
    };
    let api_key = std::env::var(key_var)
        .with_context(|| format!("missing required env var {key_var}"))?;
    let base_url = std::env::var(base_var).unwrap_or_else(|_| default_base.to_string());
    let model = std::env::var(model_var).unwrap_or_else(|_| default_model.to_string());
    Ok(LlmConfig {
        provider,
        api_key,
        base_url,
        model,
    })
}
