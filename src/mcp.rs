use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, RawContent, Tool},
    transport::TokioChildProcess,
};
use tokio::process::Command;

pub struct RudofMcp {
    service: rmcp::service::RunningService<rmcp::RoleClient, ()>,
}

impl RudofMcp {
    /// Spawn `rudof mcp` and complete the MCP initialize handshake.
    pub async fn spawn(bin: &PathBuf) -> Result<Self> {
        let mut cmd = Command::new(bin);
        cmd.arg("mcp");
        let transport = TokioChildProcess::new(cmd)
            .with_context(|| format!("spawning `{} mcp`", bin.display()))?;
        let service = ()
            .serve(transport)
            .await
            .context("MCP initialize handshake failed")?;
        Ok(Self { service })
    }

    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        self.service
            .peer()
            .list_all_tools()
            .await
            .context("MCP tools/list failed")
    }

    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let obj = arguments
            .as_object()
            .cloned()
            .unwrap_or_default();
        let params = CallToolRequestParams::new(name.to_string()).with_arguments(obj);
        let res = self
            .service
            .peer()
            .call_tool(params)
            .await
            .with_context(|| format!("call_tool({name})"))?;

        let mut text_parts: Vec<String> = res
            .content
            .iter()
            .filter_map(|c| match &c.raw {
                RawContent::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect();

        if let Some(sc) = &res.structured_content {
            text_parts.push(format!(
                "\n[structured_content]\n{}",
                serde_json::to_string_pretty(sc).unwrap_or_else(|_| sc.to_string())
            ));
        }

        if text_parts.is_empty() {
            return Err(anyhow!("MCP tool {name} returned no content"));
        }
        let combined = text_parts.join("\n");
        if res.is_error.unwrap_or(false) {
            return Err(anyhow!("MCP tool {name} returned error: {combined}"));
        }
        Ok(combined)
    }

    pub async fn shutdown(self) -> Result<()> {
        let _ = self.service.cancel().await;
        Ok(())
    }

    pub async fn validate_shacl_clean(
        &self,
        shapes_path: &str,
        result_format: &str,
    ) -> Result<String> {
        let params = CallToolRequestParams::new("validate_shacl").with_arguments(
            serde_json::json!({
                "shapes": shapes_path,
                "shapes_format": "turtle",
                "result_format": result_format,
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
        );
        let res = self
            .service
            .peer()
            .call_tool(params)
            .await
            .context("call_tool(validate_shacl)")?;
        if res.is_error.unwrap_or(false) {
            return Err(anyhow!("MCP validate_shacl returned error"));
        }
        let sc = res
            .structured_content
            .ok_or_else(|| anyhow!("validate_shacl returned no structured_content"))?;
        sc.get("results")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("validate_shacl structured_content has no `results` field"))
    }

    pub async fn export_rdf_data(&self, format: &str) -> Result<String> {
        let params = CallToolRequestParams::new("export_rdf_data").with_arguments(
            serde_json::json!({ "format": format })
                .as_object()
                .cloned()
                .unwrap_or_default(),
        );
        let res = self
            .service
            .peer()
            .call_tool(params)
            .await
            .context("call_tool(export_rdf_data)")?;
        if res.is_error.unwrap_or(false) {
            return Err(anyhow!("MCP export_rdf_data returned error"));
        }
        let sc = res
            .structured_content
            .ok_or_else(|| anyhow!("export_rdf_data returned no structured_content"))?;
        sc.get("data")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("export_rdf_data structured_content has no `data` field"))
    }
}
