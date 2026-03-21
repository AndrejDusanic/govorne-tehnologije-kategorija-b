from __future__ import annotations

import argparse
import subprocess
import sys


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def infer_username() -> str:
    email_result = run_git("config", "--get", "user.email", check=False)
    email = email_result.stdout.strip()
    if "@" in email:
        return email.split("@", 1)[0]
    name_result = run_git("config", "--get", "user.name", check=False)
    name = name_result.stdout.strip().lower().replace(" ", "-")
    return name or "your-github-username"


def current_branch() -> str:
    result = run_git("branch", "--show-current", check=False)
    branch = result.stdout.strip()
    return branch or "master"


def remote_exists(remote_name: str) -> bool:
    result = run_git("remote", "get-url", remote_name, check=False)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure the GitHub remote for this repository.")
    parser.add_argument("--username", default="", help="GitHub username")
    parser.add_argument("--repo-name", default="govorne-tehnologije-kategorija-b", help="GitHub repository name")
    parser.add_argument("--remote-name", default="origin", help="Remote name to configure")
    parser.add_argument("--branch", default="", help="Branch to push, defaults to current branch")
    parser.add_argument("--push", action="store_true", help="Push the current branch after configuring the remote")
    parser.add_argument("--print-only", action="store_true", help="Only print the inferred URL")
    args = parser.parse_args()

    username = args.username.strip() or infer_username()
    branch = args.branch.strip() or current_branch()
    remote_url = f"https://github.com/{username}/{args.repo_name}.git"
    repo_url = remote_url.removesuffix(".git")

    print(f"GitHub repo URL: {repo_url}")
    print(f"Git remote URL:  {remote_url}")
    print("Create this repository on GitHub as PRIVATE before pushing.")

    if args.print_only:
        return 0

    if remote_exists(args.remote_name):
        run_git("remote", "set-url", args.remote_name, remote_url)
        print(f"Updated remote '{args.remote_name}'.")
    else:
        run_git("remote", "add", args.remote_name, remote_url)
        print(f"Added remote '{args.remote_name}'.")

    if args.push:
        print(f"Pushing branch '{branch}'...")
        push_result = subprocess.run(["git", "push", "-u", args.remote_name, branch], text=True)
        return push_result.returncode

    print(f"Remote configured. To push later run: git push -u {args.remote_name} {branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

