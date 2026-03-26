"""
Deploy Locust master to TrueFoundry.
Master coordinates workers and serves the Web UI on port 8089.
"""

import os

from truefoundry.deploy import (
    Build,
    DockerFileBuild,
    LocalSource,
    Port,
    Resources,
    Service,
)

WORKSPACE_FQN = "tfy-usea1-devtest:hrithik-t-1"
LOCUST_AUTH_TOKEN = os.environ.get("LOCUST_AUTH_TOKEN", "")
EXPECT_WORKERS = 10
# Set TFY_SERVICE_HOST to a host configured in your cluster (Integrations > Clusters)
# to expose the Web UI publicly. If unset, port 8089 is not exposed (use port-forward).
SERVICE_HOST = os.environ.get("TFY_SERVICE_HOST", "locust-hr-master-hrithik-t-1-8089.tfy-usea1-ctl.devtest.truefoundry.tech")


def main() -> None:
    if not WORKSPACE_FQN:
        raise SystemExit("Set TFY_WORKSPACE_FQN (e.g. in .env) before deploying.")

    service = Service(
        name="locust-hr-master",
        image=Build(
            build_source=LocalSource(project_root_path="."),
            build_spec=DockerFileBuild(
                dockerfile_path="./Dockerfile",
                build_context_path=".",
                command=f"locust --master --master-bind-port 5557 --expect-workers {EXPECT_WORKERS}",
            ),
        ),
        resources=Resources(
            cpu_request=1.0,
            cpu_limit=2.0,
            memory_request=2000,
            memory_limit=4000,
            ephemeral_storage_request=1000,
            ephemeral_storage_limit=2000,
        ),
        ports=[
            Port(
                port=8089,
                protocol="TCP",
                expose=True,
                app_protocol="http",
                host="locust-hr-master-hrithik-t-1-8089.tfy-usea1-ctl.devtest.truefoundry.tech",
            ),
            Port(port=5557, protocol="TCP", expose=False, app_protocol="http"),
        ],
        workspace_fqn="tfy-usea1-devtest:hrithik-t-1",
        replicas=1.0,
            env={"LOCUST_AUTH_TOKEN": LOCUST_AUTH_TOKEN},
        )
    service.deploy(workspace_fqn=WORKSPACE_FQN)
    print("Locust master deployed. Deploy workers with: python deploy_workers.py")


if __name__ == "__main__":
    main()
