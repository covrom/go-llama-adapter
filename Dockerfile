# syntax=docker/dockerfile:1

# Build stage: static binary from the module source. The project is
# stdlib-only, so no dependency download step is needed beyond the
# module cache, which go:download handles for a single go.mod.
FROM golang:1.27-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY llama/ llama/
COPY cmd/ cmd/
RUN go build -trimpath -ldflags="-s -w" -o /out/llama-gateway ./cmd/llama-gateway

# Runtime stage: scratch keeps the image to just the binary. The
# gateway listens on :8090 by default (LLAMA_GW_LISTEN) and speaks
# the OpenAI chat-completions protocol.
FROM scratch
COPY --from=build /out/llama-gateway /llama-gateway
# busybox wget (static) is the only tool we add, so HEALTHCHECK can
# hit /healthz from inside the image.
COPY --from=build /usr/bin/wget /bin/wget
USER 65534:65534
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["/bin/wget", "-qO-", "http://127.0.0.1:8090/healthz"]
ENTRYPOINT ["/llama-gateway"]
