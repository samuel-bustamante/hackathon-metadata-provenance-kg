use std::path::Path;

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::{
    config::Config,
    llm::{LlmClient, extract_turtle_block, system_prompt, user_prompt},
    mcp::RudofMcp,
};

pub async fn run(cfg: Config) -> Result<()> {
    let ontology_ttl = std::fs::read_to_string(&cfg.ontology).context("reading ontology")?;
    let corpus_text = std::fs::read_to_string(&cfg.corpus).context("reading corpus")?;

    let abs_ontology = std::fs::canonicalize(&cfg.ontology).context("canonicalize ontology")?;
    let abs_shapes = std::fs::canonicalize(&cfg.shapes).context("canonicalize shapes")?;
    let abs_corpus = std::fs::canonicalize(&cfg.corpus).context("canonicalize corpus")?;
    let document_iri = format!("file://{}", abs_corpus.to_string_lossy());

    ensure_parent_dir(&cfg.out_kg)?;
    ensure_parent_dir(&cfg.out_report)?;

    let mcp = RudofMcp::spawn(&cfg.rudof_bin)
        .await
        .context("spawning rudof mcp")?;

    mcp.call_tool(
        "load_rdf_data_from_sources",
        serde_json::json!({
            "data": [abs_ontology.to_string_lossy()],
            "data_format": "turtle",
        }),
    )
    .await
    .context("pre-loading ontology into MCP")?;
    info!(ontology = %abs_ontology.display(), "ontology pre-loaded");

    let mcp_tools = mcp.list_tools().await.context("listing MCP tools")?;

    let llm = LlmClient::new(&cfg.llm);
    info!(
        model = llm.model_id(),
        max_tool_calls = cfg.max_tool_calls,
        "starting extraction"
    );

    let sys = system_prompt(&abs_shapes.to_string_lossy());
    let usr = user_prompt(&corpus_text, &ontology_ttl, llm.model_id(), &document_iri);

    let result = llm
        .extract_with_validation(&sys, &usr, &mcp, &mcp_tools, cfg.max_tool_calls)
        .await
        .context("LLM extraction loop")?;
    info!(tool_calls = result.tool_calls_used, "LLM loop done");

    let final_turtle = extract_turtle_block(&result.final_text);
    if final_turtle.trim().is_empty() {
        warn!("LLM final message had no Turtle; falling back to MCP export_rdf_data");
        let exported = mcp
            .export_rdf_data("turtle")
            .await
            .context("export_rdf_data fallback")?;
        std::fs::write(&cfg.out_kg, &exported)?;
    } else {
        std::fs::write(&cfg.out_kg, &final_turtle)?;
    }
    info!(path = %cfg.out_kg.display(), "wrote extracted KG");

    let _ = mcp.shutdown().await;

    let verifier = RudofMcp::spawn(&cfg.rudof_bin)
        .await
        .context("spawning rudof mcp for final verification")?;

    verifier
        .call_tool(
            "load_rdf_data_from_sources",
            serde_json::json!({
                "data": [cfg.out_kg.to_string_lossy()],
                "data_format": "turtle",
            }),
        )
        .await
        .context("loading out_kg into verifier")?;
    let report = verifier
        .validate_shacl_clean(&abs_shapes.to_string_lossy(), "turtle")
        .await
        .context("final validate_shacl")?;
    let _ = verifier.shutdown().await;

    std::fs::write(&cfg.out_report, &report)?;
    info!(path = %cfg.out_report.display(), "wrote final conformance report");

    Ok(())
}

fn ensure_parent_dir(p: &Path) -> Result<()> {
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}
