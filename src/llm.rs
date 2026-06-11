use anyhow::{Context, Result, anyhow};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionTool, ChatCompletionToolType, CreateChatCompletionRequestArgs,
        FunctionObject,
    },
};
use rmcp::model::Tool as McpTool;
use tracing::{debug, info, warn};

use crate::{config::LlmConfig, mcp::RudofMcp};

pub struct LlmClient {
    client: Client<OpenAIConfig>,
    model: String,
}

pub struct ExtractionResult {
    pub final_text: String,
    pub tool_calls_used: u32,
}

impl LlmClient {
    pub fn new(cfg: &LlmConfig) -> Self {
        let oai = OpenAIConfig::new()
            .with_api_key(&cfg.api_key)
            .with_api_base(&cfg.base_url);
        Self {
            client: Client::with_config(oai),
            model: cfg.model.clone(),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// Run the tool-calling loop. `mcp_tools` are the MCP server's tools, re-shaped to OpenAI function-tool schemas. Every model tool_call is
    /// forwarded to `mcp.call_tool` by name.
    pub async fn extract_with_validation(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        mcp: &RudofMcp,
        mcp_tools: &[McpTool],
        max_tool_calls: u32,
    ) -> Result<ExtractionResult> {
        let tools: Vec<ChatCompletionTool> =
            mcp_tools.iter().map(mcp_tool_to_openai).collect();
        info!(
            tool_count = tools.len(),
            "exposing MCP tools to LLM: {}",
            tools.iter().map(|t| t.function.name.as_str()).collect::<Vec<_>>().join(", ")
        );

        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_prompt)
                .build()
                .context("building system message")?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build()
                .context("building user message")?
                .into(),
        ];

        let mut tool_call_count = 0_u32;

        loop {
            let req = CreateChatCompletionRequestArgs::default()
                .model(&self.model)
                .temperature(0.0)
                .tools(tools.clone())
                .messages(messages.clone())
                .build()
                .context("building chat completion request")?;

            let resp = self
                .client
                .chat()
                .create(req)
                .await
                .context("chat completion request")?;
            let choice = resp
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("LLM returned zero choices"))?;
            let msg = choice.message;

            let tool_calls = msg.tool_calls.clone().unwrap_or_default();
            if tool_calls.is_empty() {
                let final_text = msg.content.clone().unwrap_or_default();
                info!(tool_calls = tool_call_count, "LLM produced final answer");
                return Ok(ExtractionResult {
                    final_text,
                    tool_calls_used: tool_call_count,
                });
            }

            let assistant_msg = ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(tool_calls.clone())
                .build()
                .context("building assistant tool_calls message")?;
            messages.push(assistant_msg.into());

            for tc in &tool_calls {
                tool_call_count += 1;
                if tool_call_count > max_tool_calls {
                    warn!(max_tool_calls, "tool-call budget exhausted; aborting LLM loop");
                    return Ok(ExtractionResult {
                        final_text: String::new(),
                        tool_calls_used: tool_call_count - 1,
                    });
                }

                let result_text = match forward_to_mcp(tc, mcp).await {
                    Ok(text) => text,
                    Err(e) => format!("ERROR running tool {}: {e:#}", tc.function.name),
                };

                let tool_msg = ChatCompletionRequestToolMessageArgs::default()
                    .content(result_text)
                    .tool_call_id(tc.id.clone())
                    .build()
                    .context("building tool message")?;
                messages.push(tool_msg.into());
            }
        }
    }
}

fn mcp_tool_to_openai(t: &McpTool) -> ChatCompletionTool {
    let parameters = serde_json::Value::Object((*t.input_schema).clone());
    ChatCompletionTool {
        r#type: ChatCompletionToolType::Function,
        function: FunctionObject {
            name: t.name.to_string(),
            description: t.description.as_ref().map(|d| d.to_string()),
            parameters: Some(parameters),
            strict: None,
        },
    }
}

async fn forward_to_mcp(tc: &ChatCompletionMessageToolCall, mcp: &RudofMcp) -> Result<String> {
    debug!(tool = %tc.function.name, "forwarding tool call to MCP");
    let args: serde_json::Value = if tc.function.arguments.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(&tc.function.arguments)
            .with_context(|| format!("parsing arguments for {}", tc.function.name))?
    };
    mcp.call_tool(&tc.function.name, args).await
}

pub fn extract_turtle_block(raw: &str) -> String {
    if let Some(start) = raw.find("```turtle") {
        let after = &raw[start + "```turtle".len()..];
        if let Some(end) = after.find("```") {
            return after[..end].trim_start_matches('\n').trim_end().to_string();
        }
    }
    if let Some(start) = raw.find("```") {
        let after = &raw[start + 3..];
        let after = after.split_once('\n').map(|(_, rest)| rest).unwrap_or(after);
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    raw.trim().to_string()
}

pub fn system_prompt(shapes_path: &str) -> String {
    SYSTEM_PROMPT_TEMPLATE.replace("{shapes_path}", shapes_path)
}

pub fn user_prompt(text: &str, kg: &str, model_id: &str, document_iri: &str) -> String {
    USER_PROMPT_TEMPLATE
        .replace("{text}", text)
        .replace("{kg}", kg)
        .replace("{model_id}", model_id)
        .replace("{document_iri}", document_iri)
}

const SYSTEM_PROMPT_TEMPLATE: &str = r#"You are a knowledge graph extractor. You are connected to the rudof MCP server, which exposes RDF tooling to you directly (`load_rdf_data_from_sources`, `validate_shacl`, `execute_sparql_query`, `export_rdf_data`, …). The existing knowledge graph has ALREADY been pre-loaded into the MCP datastore; do not re-load it.

REQUIRED WORKFLOW (do not skip)
  1. Read the user message in full.
  2. Compose your candidate Turtle (RDF 1.2 syntax with `{| ... |}` annotations).
  3. Call `load_rdf_data_from_sources` with `{ "data": ["<your candidate Turtle as ONE string>"], "data_format": "turtle" }` to push your triples into the MCP store. The datastore is CUMULATIVE — each load merges into the existing store; plan corrections as additions.
  4. Call `validate_shacl` with `{ "shapes": "{shapes_path}", "shapes_format": "turtle", "result_format": "details" }`.
  5. Read the report. If it lists violations, fix them and load corrections, then re-validate. Repeat.
  6. ONLY when `validate_shacl` reports `sh:conforms true` (no violations) may you produce a final answer. Your final answer MUST be exactly one ```turtle ... ``` fenced block containing the full Turtle output, with no commentary outside the fence.

Do not emit a final answer before `validate_shacl` reports conformance.
"#;

const USER_PROMPT_TEMPLATE: &str = r#"
You are given:
1. A source text containing statements about food items and their suitability for different diets.
2. An existing RDF knowledge graph containing the allowed ontology vocabulary.
3. An RDF 1.2 provenance metadata pattern and a SHACL shape for validating that metadata.

## Input
### Source text
{text}

### Existing knowledge graph
{kg}

### Document IRI (use this value verbatim in every `ctx:extractedFrom`)
<{document_iri}>

### RDF 1.2 provenance metadata pattern
focosa-f:GrilledSalmon
    schema:suitableForDiet focosa-h:PescatarianDiet
    {|
        llm:LLM_id "{model_id}" ;
        llm:validationMethod "shacl" ;
        ctx:extractedFrom <{document_iri}>
    |} .

### SHACL metadata shape
shapes:FoodItemShape
    a sh:NodeShape ;
    sh:targetClass focosa-f:FoodItem ;
    sh:property [
        sh:path schema:suitableForDiet ;
        sh:nodeKind sh:IRI ;
        sh:reifierShape shapes:ProvenanceShape ;
        sh:reificationRequired true
    ] .

shapes:ProvenanceShape
    a sh:NodeShape ;
    sh:property [ sh:path llm:LLM_id;           sh:minCount 1; sh:maxCount 1; sh:datatype xsd:string ] ;
    sh:property [ sh:path llm:validationMethod; sh:minCount 1; sh:maxCount 1; sh:datatype xsd:string ] ;
    sh:property [ sh:path ctx:extractedFrom;    sh:minCount 1; sh:maxCount 1; sh:nodeKind sh:IRI ] .

## Task
Start from the existing knowledge graph and enrich only its A-Box.
Extract every food item and every supported factual relation that can be derived from the complete source text.
Before generating a new individual or assertion, check whether it already exists in the supplied knowledge graph. Do not duplicate existing individuals 
or triples.

## Strict ontology constraints
1. Do not modify the ontology.
2. Do not add any new class.
3. Do not add any new subclass.
4. Do not add any new domain or range declaration.
5. Do not add any new diet type, allergy type, food category, health condition, or other concept.
6. Do not invent predicates.
7. Use only classes, properties, and controlled concepts already present in the supplied knowledge graph.
8. Preserve the exact IRIs used in the knowledge graph, including their spelling. Do not correct, rename, normalise, or replace existing IRIs.
9. Generate only individuals and instance-level assertions.
10. Do not declare a diet individual as an instance of focosa-h:RestrictedDiet. Diet resources already defined in the ontology must only be referenced.
11. Do not use external Schema.org diet resources unless those exact resources already occur in the supplied knowledge graph.
12. Do not infer unsupported facts from general world knowledge. Use only information explicitly stated in the source text.
13. Do not represent negative statements unless the existing knowledge graph already contains an appropriate predicate for them.
14. The absence of a suitability assertion must not be represented as an explicit negative claim.

For example, this is prohibited:
focosa-h:HalalDiet
    a focosa-h:RestrictedDiet .

It adds ontology-level information and is not an instance assertion about a food item.

## Food individuals
For every food item described in the text:
1. Reuse its existing IRI when the food already exists in the knowledge graph.
2. Otherwise, create a new IRI in the focosa-f: namespace.
3. Type it only with an existing class such as focosa-f:FoodItem.
4. Add its exact human-readable name using rdfs:label.
5. Add only relations whose predicates and objects already exist in the knowledge graph.

Example structure:
focosa-f:BrownRiceBoiled
    a focosa-f:FoodItem ;
    rdfs:label "Brown rice, boiled" .

## Provenance metadata
Every newly generated factual assertion must include RDF 1.2 triple annotation metadata.
For every extracted assertion, attach exactly these three metadata fields:
{|
    llm:LLM_id "{model_id}" ;
    llm:validationMethod "shacl" ;
    ctx:extractedFrom <{document_iri}>
|}

Use the Document IRI value <{document_iri}> verbatim as the object of every `ctx:extractedFrom`. Do not invent new document IRIs.

Example annotated assertion:
focosa-f:BrownRiceBoiled
    schema:suitableForDiet focosa-h:GlutenFreeDiet
    {|
        llm:LLM_id "{model_id}" ;
        llm:validationMethod "shacl" ;
        ctx:extractedFrom <{document_iri}>
    |} .

Metadata predicates may be used only as defined by the supplied provenance pattern and SHACL shape. Do not introduce additional metadata predicates (no `prov:wasGeneratedBy`, no `llm:groundedIn`, no `llm:TextChunk`).

## Extraction rules
Process the entire source text, not only the first example.
For each food item:
1. Extract every explicitly stated positive diet-suitability relation.
2. Retain an assertion only when:
  - the property already exists in the knowledge graph, and
  - the diet resource already exists in the knowledge graph.
3. Omit unsupported relations rather than creating new vocabulary.
4. Do not encode food categories unless the existing knowledge graph already contains a suitable property and category resource.
5. Do not encode ingredients, allergens, chronic conditions, glycaemic properties, or avoidance recommendations unless the required predicates and objects already exist in the supplied knowledge graph.
6. Do not duplicate assertions already present in the knowledge graph.
7. When two similar-looking IRIs exist, use only the exact IRI that is declared or already used consistently in the supplied ontology. Do not assume that differently spelled IRIs are equivalent.

## Output requirements
Return only valid Turtle-star syntax using RDF 1.2 triple annotations.
The output must contain only:
- newly required food individuals;
- their rdf:type assertions;
- their rdfs:label assertions;
- valid instance-level relations derived from the text;
- RDF 1.2 provenance annotations for every newly extracted factual relation.

Do not output:
- explanations;
- summaries;
- warnings;
- markdown commentary;
- ontology declarations;
- class declarations;
- property declarations;
- subclass axioms;
- duplicate triples;
- unsupported facts;
- invented vocabulary.

Place the complete Turtle output inside one code block.
"#;
