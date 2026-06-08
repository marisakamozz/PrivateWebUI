Do not run `terraform apply`, `terraform destroy`, or post-apply Docker/runtime status checks unless the user explicitly asks for them. Provide the user with the commands to run instead.
For AWS SSM Run Command via the AWS CLI, write the `--parameters` payload to a temporary JSON file under `/private/tmp` and pass it with `--parameters file://...` to avoid shell quoting issues. Regular cleanup is not required because these temporary files are removed on macOS restart.
Run AWS SSM CLI operations with escalated permissions from the start instead of the local sandbox.
