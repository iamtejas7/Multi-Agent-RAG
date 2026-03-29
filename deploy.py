"""
Deploy the Enterprise Knowledge Assistant to AWS Bedrock AgentCore Runtime
using the agentcore CLI (bedrock-agentcore-starter-toolkit).

Prerequisites:
    pip install bedrock-agentcore-starter-toolkit

Steps:
    1. Configure:  agentcore configure -e agent.py --name technova-assistant --disable-memory --disable-otel -dt direct_code_deploy -rt PYTHON_3_13 -rf requirements.txt
    2. Deploy:     agentcore deploy
    3. Test:       agentcore invoke '{"prompt": "How many annual leaves do I get?"}'
    4. Status:     agentcore status
    5. Cleanup:    agentcore destroy

Or run this script to execute steps 1-2 automatically:
    python deploy.py [--region us-east-1] [--name technova-assistant]
"""
import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print(f"\n> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Deploy to AgentCore Runtime via CLI")
    parser.add_argument("--region", "-r", default="us-east-2", help="AWS region")
    parser.add_argument("--name", "-n", default="technova-assistant", help="Agent name")
    args = parser.parse_args()

    # Step 1: Configure
    run([
        "agentcore", "configure",
        "--entrypoint", "agent.py",
        "--name", args.name,
        "--requirements-file", "requirements-agentcore.txt",
        "--deployment-type", "direct_code_deploy",
        "--runtime", "PYTHON_3_13",
        "--region", args.region,
        "--disable-memory",
        "--disable-otel",
        "--non-interactive",
    ])

    # Step 2: Deploy
    run(["agentcore", "deploy"])

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)
    print(f"\nTest with:")
    print(f'  agentcore invoke \'{{"prompt": "How many annual leaves do I get?"}}\'\n')
    print(f"Check status:   agentcore status")
    print(f"Stop session:   agentcore stop-session")
    print(f"Destroy:        agentcore destroy")


if __name__ == "__main__":
    main()
