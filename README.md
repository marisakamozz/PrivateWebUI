# PrivateWebUI on AWS

Terraform for a private EC2-based LibreChat + Open WebUI + vLLM stack.

## Architecture

- EC2 runs Docker Compose in a private subnet with no public IP.
- vLLM exposes an OpenAI-compatible API on instance-local port `8000`.
- LibreChat exposes the web UI on instance-local port `3080`.
- Open WebUI exposes the web UI on instance-local port `3000`.
- Access uses AWS Systems Manager Session Manager port forwarding.
- SSM, SSM messages, EC2 messages, Secrets Manager, CloudWatch Logs, and S3 have VPC endpoints.
- A NAT Gateway is enabled by default so the private instance can pull Docker images and Hugging Face model files.
- The Hugging Face token is stored in AWS Secrets Manager. Terraform creates the secret, but you set the secret value.

## Deploy

```bash
terraform init
terraform apply
```

Set your Hugging Face token after the secret is created:

```bash
aws secretsmanager put-secret-value \
  --region us-west-2 \
  --secret-id private-web-ui/huggingface-token \
  --secret-string '<your-hf-token>'
```

Restart the app so the instance rereads the secret:

```bash
aws ssm start-session --region us-west-2 --target "$(terraform output -raw instance_id)"
sudo systemctl restart private-web-ui
```

Wait for the containers to start from inside the EC2 instance:

```bash
aws ssm start-session --region us-west-2 --target "$(terraform output -raw instance_id)"
watch -n 5 'sudo systemctl status private-web-ui --no-pager; echo; cd /opt/private-web-ui && sudo docker compose ps'
```

After the containers are running, follow the vLLM logs and wait for startup to
finish:

```bash
cd /opt/private-web-ui
sudo docker compose logs -f vllm
```

vLLM is ready when the log prints `Application startup complete.`.

Forward LibreChat to your local machine:

```bash
aws ssm start-session \
  --region us-west-2 \
  --target "$(terraform output -raw instance_id)" \
  --document-name AWS-StartPortForwardingSession \
  --parameters file://aws/ssm-parameters/librechat-port-forward.json
```

Then open `http://localhost:3080`.

Forward Open WebUI to your local machine:

```bash
aws ssm start-session \
  --region us-west-2 \
  --target "$(terraform output -raw instance_id)" \
  --document-name AWS-StartPortForwardingSession \
  --parameters file://aws/ssm-parameters/open-webui-port-forward.json
```

Then open `http://localhost:3000`.

Optionally forward vLLM directly:

```bash
aws ssm start-session \
  --region us-west-2 \
  --target "$(terraform output -raw instance_id)" \
  --document-name AWS-StartPortForwardingSession \
  --parameters file://aws/ssm-parameters/vllm-port-forward.json
```

## Notes

- Default instance type is `g6e.xlarge`, which provides one NVIDIA L40S 44 GiB GPU.
- Default model is `google/gemma-4-31B-it-qat-w4a16-ct` with Gemma 4 MTP speculative decoding enabled via `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`.
- LibreChat email login and registration are enabled for the initial development setup.
- Open WebUI uses the same vLLM OpenAI-compatible endpoint, the same SearXNG container for Web Search, and browser-side Pyodide for code execution and code interpreter. After first login, enable the model's Code Interpreter capability from the Open WebUI admin model settings if it is not already enabled.
- Open WebUI reasoning controls are available from Chat Controls or model Advanced Parameters. For Gemma 4 on vLLM, set `reasoning_effort` to values such as `low`, `medium`, or `high` when you want to enable/tune thinking.
- If you require no internet egress at all, disable the NAT Gateway and pre-stage Docker images and model artifacts through private channels first.
