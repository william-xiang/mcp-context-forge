# MCP Context Forge - OpenShift Deployment on IBM Power

This directory contains Kubernetes/OpenShift manifests for deploying the MCP Context Forge application using Kustomize.

## Overview

MCP Context Forge is deployed on OpenShift using a source-to-image (S2I) build strategy. With this strategy, the application can be built from source in a remote repository or local development environment. The deployment of MCP Context Forge is configured with an ImageStreamTag trigger; any new image build will trigger the application re-deployment.

## Prerequisites

1. OpenShift cluster access
2. `oc` CLI tool installed
3. Kustomize installed
4. A `.env` file in the `mcp-context-forge/` directory with required environment variables

## Deployment Instructions

1. **Clone the repository**

    ```bash
    git clone https://github.com/william-xiang/mcp-context-forge.git -b openshift_deployment
    cd mcp-context-forge/deployment/openshift
    ```

    > Note: Due to an image build issue for IBM Power, we use a fork of the original repo for now.
    Once this [PR](https://github.com/IBM/mcp-context-forge/pull/1466) is merged, we can use the original repo again.

2. **Set your namespace**

    Replace `<NAMESPACE_PLACEHOLDER>` in the root `kustomization.yaml` with your actual OpenShift namespace.

3. **Create the `.env` file**

    Run the command below to create a `.env` file in the `mcp-context-forge/` directory with your application configuration. You can keep the default values as they are.

    ```bash
    cp ../../.env.example ./mcp-context-forge/.env
    ```

4. **Deploy using Kustomize**:
   ```bash
   kustomize build . | oc apply -f -
   ```

5. **Monitor the image build**:
   ```bash
   oc logs -f bc/mcp-context-forge
   ```

6. **Check deployment status**:
   ```bash
   oc get pods
   oc get route mcp-context-forge
   ```

7. **Trigger the build manually**:
   ```bash
   # Trigger a new build using the source from remote repo
   oc start-build mcp-context-forge
   # Trigger a new build using the source from local
   oc start-build mcp-context-forge --from-dir=.
   ```

## Notes

The build config specifies to use the branch `rust_base_image` of the repo `https://github.com/william-xiang/mcp-context-forge.git` in `mcp-context-forge/mcp-context-forge-buildconfig.yaml`. This branch contains the fix for the build issue on Power. You can change these two values to use another repo or branch.
After changes are made in this file, you can run the command `kustomize build . | oc apply -f -` to apply the changes. Then a new build will be triggered and eventually the application will be redeployed using the new image.


## Troubleshooting

- **Build failures**: Check build logs with `oc logs -f bc/mcp-context-forge`
- **Pod crashes**: Check pod logs with `oc logs <pod-name>`
- **Storage issues**: Verify PVC is bound with `oc get pvc`
- **Access issues**: Verify route with `oc get route mcp-context-forge`