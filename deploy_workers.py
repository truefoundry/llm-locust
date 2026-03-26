"""
Deploy Locust workers to TrueFoundry with HPA autoscaling.
Workers connect to the master via internal service name locust-hr-master:5557.
"""

import os

from truefoundry.deploy import (
    Build,
    DockerFileBuild,
    LocalSource,
    NodeSelector,
    Port,
    Resources,
    Service,
)

WORKSPACE_FQN = "tfy-usea1-devtest:hrithik-t-1"
LOCUST_AUTH_TOKEN = os.environ.get("LOCUST_AUTH_TOKEN", "")


def main() -> None:
    if not WORKSPACE_FQN:
        raise SystemExit("Set TFY_WORKSPACE_FQN (e.g. in .env) before deploying.")

    service = Service(
        name="locust-hr-worker",
        image=Build(
            build_source=LocalSource(project_root_path="."),
            build_spec=DockerFileBuild(
                dockerfile_path="./Dockerfile",
                build_context_path=".",
                command="locust --worker --master-host locust-hr-master --master-port 5557",
            ),
        ),
        resources=Resources(
        cpu_request=2.0,
        cpu_limit=4.0,
        memory_request=4000,
        memory_limit=6000,
        ephemeral_storage_request=10000,
        ephemeral_storage_limit=20000,
        node=NodeSelector(capacity_type="spot_fallback_on_demand"),
        ),
        ports=[Port(port=8089, protocol="TCP", expose=False, app_protocol="http")],
        workspace_fqn="tfy-usea1-devtest:hrithik-t-1",
        replicas=100.0,
        env={"LOCUST_AUTH_TOKEN": LOCUST_AUTH_TOKEN},
    )
    service.deploy(workspace_fqn=WORKSPACE_FQN)
    print("Locust workers deployed. HPA will scale 10–30 replicas at 70% CPU.")


if __name__ == "__main__":
    main()
