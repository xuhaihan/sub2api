package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/Wei-Shaw/sub2api/internal/config"
)

func TestBuildDeepSeekCompactRequestBody(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"stream":true,
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
			{"type":"compaction_trigger"}
		],
		"tools":[{"type":"function","name":"shell"}]
	}`)

	patched, err := buildDeepSeekCompactRequestBody(body)
	require.NoError(t, err)
	require.False(t, gjson.GetBytes(patched, "stream").Bool())
	require.False(t, gjson.GetBytes(patched, "store").Bool())
	require.Equal(t, "none", gjson.GetBytes(patched, "tool_choice").String())
	require.False(t, gjson.GetBytes(patched, "include").Exists(), "DeepSeek 不支持 include")

	input := gjson.GetBytes(patched, "input").Array()
	require.Len(t, input, 2, "compaction_trigger 应被丢弃，仅保留原消息与摘要指令")
	require.Equal(t, "message", input[0].Get("type").String())
	require.Equal(t, "message", input[1].Get("type").String())
	prompt := input[1].Get("content.0.text").String()
	require.Contains(t, prompt, "1. Primary Request and Intent")
	require.Contains(t, prompt, "Respond with ONLY the <summary>...</summary> block")
}

func TestBuildDeepSeekCompactRequestBodyConvertsPriorSummary(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"stream":true,
		"input":[
			{"id":"cmp_old","type":"compaction","status":"completed","summary":[{"type":"summary_text","text":"earlier summary"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]},
			{"type":"compaction_trigger"}
		]
	}`)

	patched, err := buildDeepSeekCompactRequestBody(body)
	require.NoError(t, err)
	input := gjson.GetBytes(patched, "input").Array()
	require.Len(t, input, 3)
	require.Equal(t, "message", input[0].Get("type").String())
	require.Contains(t, input[0].Get("content.0.text").String(), "<conversation_summary>")
	require.Contains(t, input[0].Get("content.0.text").String(), "earlier summary")
	require.Equal(t, "message", input[1].Get("type").String())
	require.Equal(t, "message", input[2].Get("type").String())
}

func TestConvertDeepSeekResponseToOpenAICompact(t *testing.T) {
	body := []byte(`{
		"id":"resp_ds_1",
		"object":"response",
		"status":"completed",
		"model":"deepseek-v4-flash",
		"output":[
			{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"summary text"}]}
		],
		"usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120}
	}`)

	converted, err := convertDeepSeekResponseToOpenAICompact(body)
	require.NoError(t, err)
	require.Equal(t, "resp_ds_1", gjson.GetBytes(converted, "id").String())
	require.Equal(t, "completed", gjson.GetBytes(converted, "status").String())
	output := gjson.GetBytes(converted, "output").Array()
	require.Len(t, output, 1)
	require.Equal(t, "compaction", output[0].Get("type").String())
	require.Equal(t, "summary text", output[0].Get("summary.0.text").String())
	decoded, err := base64.StdEncoding.DecodeString(output[0].Get("encrypted_content").String())
	require.NoError(t, err)
	require.Equal(t, "summary text", string(decoded))
	require.Equal(t, int64(120), gjson.GetBytes(converted, "usage.total_tokens").Int())
}

func TestConvertDeepSeekResponseToOpenAICompactUsesReasoningFallback(t *testing.T) {
	body := []byte(`{
		"id":"resp_ds_2",
		"status":"completed",
		"output":[
			{"id":"rs_1","type":"reasoning","content":[{"type":"output_text","text":"thinking summary"}]}
		],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`)

	converted, err := convertDeepSeekResponseToOpenAICompact(body)
	require.NoError(t, err)
	require.Equal(t, "thinking summary", gjson.GetBytes(converted, "output.0.summary.0.text").String())
}

func TestConvertDeepSeekResponseToOpenAICompactRequiresSummary(t *testing.T) {
	body := []byte(`{"id":"resp_ds_3","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	_, err := convertDeepSeekResponseToOpenAICompact(body)
	require.ErrorContains(t, err, "no summary text")
}

func TestConvertOpenAICompactInputsForDeepSeek(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"input":[
			{"id":"cmp_1","type":"compaction","status":"completed","summary":[{"type":"summary_text","text":"compacted context"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
		]
	}`)

	converted, changed, err := convertOpenAICompactInputsForDeepSeek(body)
	require.NoError(t, err)
	require.True(t, changed)
	input := gjson.GetBytes(converted, "input").Array()
	require.Len(t, input, 2)
	require.Equal(t, "message", input[0].Get("type").String())
	require.Contains(t, input[0].Get("content.0.text").String(), "<conversation_summary>")
	require.Contains(t, input[0].Get("content.0.text").String(), "compacted context")
	require.Equal(t, "continue", input[1].Get("content.0.text").String())
}

func TestConvertOpenAICompactInputsForDeepSeekBase64Fallback(t *testing.T) {
	summary := "fallback summary"
	body := []byte(`{"model":"deepseek-v4-flash","input":[{"id":"cmp_2","type":"compaction","status":"completed","encrypted_content":"` + base64.StdEncoding.EncodeToString([]byte(summary)) + `"}]}`)

	converted, changed, err := convertOpenAICompactInputsForDeepSeek(body)
	require.NoError(t, err)
	require.True(t, changed)
	require.Contains(t, gjson.GetBytes(converted, "input.0.content.0.text").String(), summary)
}

func TestNormalizeDeepSeekResponsesRequestBodyCompactAliasAndSummary(t *testing.T) {
	account := &Account{
		Platform: PlatformDeepseek,
		Credentials: map[string]any{
			"api_protocol": APIProtocolResponses,
		},
	}
	body := []byte(`{
		"model":"deepseek-v4-flash-openai-compact",
		"previous_response_id":"resp_prev",
		"store":true,
		"input":[
			{"id":"cmp_1","type":"compaction","status":"completed","summary":[{"type":"summary_text","text":"saved"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
		]
	}`)

	normalized := normalizeDeepSeekResponsesRequestBody(account, body)
	require.Equal(t, "deepseek-v4-flash", gjson.GetBytes(normalized, "model").String())
	require.False(t, gjson.GetBytes(normalized, "store").Bool())
	require.False(t, gjson.GetBytes(normalized, "previous_response_id").Exists())
	require.Equal(t, "message", gjson.GetBytes(normalized, "input.0.type").String())
	require.Contains(t, gjson.GetBytes(normalized, "input.0.content.0.text").String(), "saved")
}

func TestOpenAICompactSupportTierDeepSeek(t *testing.T) {
	responses := &Account{
		Platform: PlatformDeepseek,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_protocol": APIProtocolResponses,
		},
		Extra: map[string]any{"openai_responses_supported": true},
	}
	require.Equal(t, 2, openAICompactSupportTier(responses))

	// 未探测也默认走 Responses（现状即证据），本地压缩链路依然可用。
	unknown := &Account{
		Platform: PlatformDeepseek,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_protocol": APIProtocolResponses,
		},
		Extra: map[string]any{},
	}
	require.Equal(t, 2, openAICompactSupportTier(unknown))

	// Chat Completions 直转账号没有本地压缩链路，保持不可用。
	chatOnly := &Account{
		Platform: PlatformDeepseek,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_protocol": APIProtocolChatCompletions,
		},
		Extra: map[string]any{"openai_responses_supported": false},
	}
	require.Equal(t, 0, openAICompactSupportTier(chatOnly))
}

func TestOpenAIGatewayService_SelectAccountWithScheduler_DeepSeekCompactEligible(t *testing.T) {
	for _, advanced := range []bool{false, true} {
		t.Run(map[bool]string{false: "legacy_scheduler", true: "advanced_scheduler"}[advanced], func(t *testing.T) {
			resetOpenAIAdvancedSchedulerSettingCacheForTest()
			svc := newOpenAICompactionSchedulerTestService([]Account{
				{
					ID:          72001,
					Platform:    PlatformDeepseek,
					Type:        AccountTypeAPIKey,
					Status:      StatusActive,
					Schedulable: true,
					Concurrency: 1,
					Credentials: map[string]any{
						"api_protocol": APIProtocolResponses,
					},
					Extra: map[string]any{},
				},
			}, advanced)

			groupID := int64(91020)
			selection, _, err := svc.SelectAccountWithSchedulerForCapability(
				context.Background(),
				&groupID,
				"",
				"",
				"deepseek-v4-flash",
				nil,
				OpenAIUpstreamTransportAny,
				OpenAIEndpointCapabilityResponses,
				true,
				false,
				false,
				PlatformDeepseek,
			)
			require.NoError(t, err)
			require.NotNil(t, selection)
			require.NotNil(t, selection.Account)
			require.Equal(t, int64(72001), selection.Account.ID)
		})
	}
}

func TestOpenAIGatewayService_Forward_DeepSeekCompactLegacy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{"model":"deepseek-v4-flash","stream":false,"instructions":"compact-test","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses/compact", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid-ds-compact"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"resp_ds_compact",
			"object":"response",
			"status":"completed",
			"model":"deepseek-v4-flash",
			"output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"summary text"}]}],
			"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14}
		}`)),
	}}

	svc := &OpenAIGatewayService{httpUpstream: upstream, cfg: &config.Config{}}
	account := deepSeekCompactTestAccount()

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "deepseek-v4-flash", result.Model)

	// 上游收到的是本地摘要回合：unary + 摘要指令。
	require.False(t, gjson.GetBytes(upstream.lastBody, "stream").Bool())
	input := gjson.GetBytes(upstream.lastBody, "input").Array()
	require.Len(t, input, 2)
	require.Contains(t, input[1].Get("content.0.text").String(), "Primary Request and Intent")

	// 下游收到 compaction 输出项。
	require.Equal(t, "compaction", gjson.GetBytes(rec.Body.Bytes(), "output.0.type").String())
	require.Equal(t, "summary text", gjson.GetBytes(rec.Body.Bytes(), "output.0.summary.0.text").String())
	require.Equal(t, int64(14), gjson.GetBytes(rec.Body.Bytes(), "usage.total_tokens").Int())
}

func TestOpenAIGatewayService_Forward_DeepSeekCompactNativeV2SSE(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"stream":true,
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
			{"type":"compaction_trigger"}
		]
	}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	MarkOpenAINativeCompactionV2(c)

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid-ds-v2"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"resp_ds_v2",
			"object":"response",
			"status":"completed",
			"model":"deepseek-v4-flash",
			"output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"v2 summary"}]}],
			"usage":{"input_tokens":20,"output_tokens":5,"total_tokens":25}
		}`)),
	}}

	svc := &OpenAIGatewayService{httpUpstream: upstream, cfg: &config.Config{}}
	account := deepSeekCompactTestAccount()

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)

	// 原生 v2 被重写为 legacy compact 路径：上游 unary、无 compaction_trigger。
	require.False(t, gjson.GetBytes(upstream.lastBody, "stream").Bool(), "stream should be false upstream")
	require.False(t, gjson.GetBytes(upstream.lastBody, "input.#(type==compaction_trigger)").Exists(), "compaction_trigger must be dropped")
	require.True(t, bytes.Contains(upstream.lastBody, []byte("Primary Request and Intent")), "summary prompt must be appended")

	// 下游按 SSE 协议回写：output_item.done 携带 compaction item + completed。
	output := rec.Body.String()
	require.Contains(t, output, "event: response.output_item.done", "SSE bridge should emit output_item.done")
	require.Contains(t, output, `"type":"compaction"`, "SSE item must be compaction")
	require.Contains(t, output, `"type":"summary_text"`, "SSE item must carry summary")
	require.Contains(t, output, "event: response.completed", "SSE bridge should emit response.completed")
	require.Contains(t, output, `"id":"resp_ds_v2"`, "completed event must keep response id")
}

func deepSeekCompactTestAccount() *Account {
	return &Account{
		ID:          73001,
		Name:        "deepseek-responses",
		Platform:    PlatformDeepseek,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":      "sk-test-deepseek",
			"base_url":     "https://api.deepseek.com",
			"api_protocol": APIProtocolResponses,
		},
		Extra:       map[string]any{"openai_responses_supported": true},
		Status:      StatusActive,
		Schedulable: true,
	}
}
