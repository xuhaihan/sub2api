package service

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/google/uuid"
)

// isDeepSeekServerSideCompactAccount 判断账号是否走"网关本地压缩"链路。
//
// DeepSeek 官方 Responses 端点（适配 Codex）不提供 /responses/compact，
// 也不识别 compaction_trigger / compaction 输入项——文档明确"Other types
// are ignored"。因此远程压缩由 sub2api 自己执行：把客户端 compact 请求转换
// 成一个普通摘要回合，再把模型回复组装成 Codex 期望的 compaction 输出项。
// 仅当账号确实走原生 Responses 协议时才启用；Chat Completions 直转账号仍
// 维持原有的 compact 不可用行为。
func isDeepSeekServerSideCompactAccount(account *Account) bool {
	return account != nil &&
		account.Platform == PlatformDeepseek &&
		account.GetAPIProtocol() == APIProtocolResponses &&
		openai_compat.ShouldUseResponsesAPI(account.Extra)
}

// buildDeepSeekCompactRequestBody 把客户端 compact 请求转换为 DeepSeek 可执行
// 的普通摘要回合：
//   - 丢弃 compaction_trigger（DeepSeek 忽略未知 input item，但保留无意义）；
//   - 把历史中的 compaction 摘要展开为 <conversation_summary> 用户消息；
//   - 追加 grokCompactSummaryPrompt 摘要指令；
//   - stream=false / store=false / tool_choice=none，使上游走 unary JSON。
//
// DeepSeek 不支持 include（静默忽略），因此不携带
// "reasoning.encrypted_content"（该平台也没有加密状态可回带）。
func buildDeepSeekCompactRequestBody(body []byte) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("decode compact request: %w", err)
	}

	input, err := normalizeGrokCompactInput(payload["input"])
	if err != nil {
		return nil, err
	}
	input, _ = convertDeepSeekCompactInputItems(input)
	input = dropCompactTriggerInputItems(input)
	input = append(input, map[string]any{
		"type": "message",
		"role": "user",
		"content": []any{map[string]any{
			"type": "input_text",
			"text": grokCompactSummaryPrompt,
		}},
	})
	payload["input"] = input
	payload["store"] = false
	payload["stream"] = false
	if tools, ok := payload["tools"].([]any); ok && len(tools) > 0 {
		payload["tool_choice"] = "none"
	}

	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode compact request: %w", err)
	}
	return encoded, nil
}

// dropCompactTriggerInputItems 移除 compaction_trigger 输入项。摘要指令本身
// 就是新的最后一项用户消息，触发项再发给 DeepSeek 没有任何意义。
func dropCompactTriggerInputItems(input []any) []any {
	out := input[:0]
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if ok && strings.TrimSpace(stringValue(item["type"])) == "compaction_trigger" {
			continue
		}
		out = append(out, raw)
	}
	return out
}

// convertDeepSeekResponseToOpenAICompact 把 DeepSeek 摘要回合的终态 JSON 组装
// 成 Codex remote compact 需要的 compaction 输出项。DeepSeek 没有
// encrypted_content：摘要文本放 summary；encrypted_content 放摘要的 base64，
// 作为客户端回带时的稳定占位（还原端优先读 summary）。
func convertDeepSeekResponseToOpenAICompact(body []byte) ([]byte, error) {
	var response map[string]any
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	output, ok := response["output"].([]any)
	if !ok {
		return nil, fmt.Errorf("response has no output array")
	}

	var summaryParts []string
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch strings.TrimSpace(stringValue(item["type"])) {
		case "message":
			summaryParts = append(summaryParts, outputItemTextParts(item["content"])...)
		case "reasoning":
			// DeepSeek 思维链明文 content 会合并进相邻 assistant 消息；若上游
			// 只返回 reasoning，则把它的 content 也视为摘要候选。
			summaryParts = append(summaryParts, outputItemTextParts(item["content"])...)
		}
	}
	summary := strings.TrimSpace(strings.Join(summaryParts, "\n"))
	if summary == "" {
		return nil, fmt.Errorf("compact response has no summary text")
	}

	compactItem := map[string]any{
		"id":                "cmp_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
		"type":              "compaction",
		"status":            "completed",
		"summary":           []any{map[string]any{"type": "summary_text", "text": summary}},
		"encrypted_content": base64.StdEncoding.EncodeToString([]byte(summary)),
	}
	response["output"] = []any{compactItem}
	response["status"] = "completed"
	delete(response, "output_text")

	encoded, err := json.Marshal(response)
	if err != nil {
		return nil, fmt.Errorf("encode compact response: %w", err)
	}
	return encoded, nil
}

func outputItemTextParts(value any) []string {
	var parts []string
	switch content := value.(type) {
	case string:
		if text := strings.TrimSpace(content); text != "" {
			parts = append(parts, text)
		}
	case []any:
		for _, rawPart := range content {
			part, ok := rawPart.(map[string]any)
			if !ok {
				continue
			}
			if text := strings.TrimSpace(stringValue(part["text"])); text != "" {
				parts = append(parts, text)
			}
		}
	}
	return parts
}

// convertOpenAICompactInputsForDeepSeek 把客户端回带的 compaction 输出项还原为
// DeepSeek 可读的 <conversation_summary> 用户消息。DeepSeek 对未知 input item
// 一律忽略，若不还原，压缩后的上下文会直接丢失。
func convertOpenAICompactInputsForDeepSeek(body []byte) ([]byte, bool, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return body, false, err
	}
	items, ok := payload["input"].([]any)
	if !ok {
		return body, false, nil
	}

	converted, changed := convertDeepSeekCompactInputItems(items)
	if !changed {
		return body, false, nil
	}
	payload["input"] = converted
	encoded, err := json.Marshal(payload)
	if err != nil {
		return body, false, err
	}
	return encoded, true, nil
}

// convertDeepSeekCompactInputItems 把 compaction 输出项还原为
// <conversation_summary> 用户消息；返回新切片与是否发生变更。
func convertDeepSeekCompactInputItems(items []any) ([]any, bool) {
	changed := false
	converted := make([]any, 0, len(items))
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok || !isOpenAICompactionType(stringValue(item["type"])) {
			converted = append(converted, raw)
			continue
		}
		changed = true
		summary := compactSummaryText(item["summary"])
		if summary == "" {
			if encoded := strings.TrimSpace(stringValue(item["encrypted_content"])); encoded != "" {
				if decoded, err := base64.StdEncoding.DecodeString(encoded); err == nil {
					summary = strings.TrimSpace(string(decoded))
				}
			}
		}
		if summary == "" {
			continue
		}
		converted = append(converted, map[string]any{
			"type": "message",
			"role": "user",
			"content": []any{map[string]any{
				"type": "input_text",
				"text": "<conversation_summary>\n" + summary + "\n</conversation_summary>",
			}},
		})
	}
	return converted, changed
}
