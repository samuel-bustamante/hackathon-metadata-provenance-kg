use std::path::PathBuf;
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "kg_augmentation_rdf1.2_shacl",
    about = "Augments a knowledge graph with LLM-extracted A-Box assertions carrying RDF 1.2 statement-level provenance, validated in a SHACL tool-calling loop via the rudof MCP server"
)]
pub struct Cli {
    #[arg(long, value_enum)]
    pub model: LlmProvider,

    #[arg(long, env = "ONTOLOGY_PATH", default_value = "inputs/gold_standard.owl")]
    pub ontology: PathBuf,

    #[arg(long, env = "SHAPES_PATH", default_value = "inputs/shapes/provenance_shapes.ttl")]
    pub shapes: PathBuf,

    #[arg(long, env = "CORPUS_PATH", default_value = "inputs/food_diet_expansion.txt")]
    pub corpus: PathBuf,

    #[arg(long, env = "OUT_KG", default_value = "outputs/extracted_kg.ttl")]
    pub out_kg: PathBuf,

    #[arg(long, env = "OUT_REPORT", default_value = "outputs/conformance_report.ttl")]
    pub out_report: PathBuf,

    #[arg(long, env = "MAX_TOOL_CALLS", default_value_t = 30)]
    pub max_tool_calls: u32,

    #[arg(long, env = "RUDOF_BIN", default_value = "/home/samu/.local/bin/rudof")]
    pub rudof_bin: PathBuf,
}

#[derive(ValueEnum, Clone, Debug, Copy)]
pub enum LlmProvider {
    Qwen,
    Deepseek,
}
