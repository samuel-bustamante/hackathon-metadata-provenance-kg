use anyhow::{Context, Result, anyhow};
use clap::Parser;

mod cli;
mod config;
mod llm;
mod mcp;
mod pipeline;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = cli::Cli::parse();
    let cfg = config::Config::from_cli(cli).context("building configuration")?;
    pipeline::run(cfg)
        .await
        .map_err(|e| anyhow!("pipeline failed: {e:#}"))
}
