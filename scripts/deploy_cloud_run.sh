#!/usr/bin/env bash
# Deployment helper script for TurbineGuard on GCP Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-asia-southeast1}"
SERVICE_NAME="turbineguard-api"
REPO_NAME="turbineguard-repo"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "=== TurbineGuard Cloud Run Deployment ==="
echo "Project ID: ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Image URI:  ${IMAGE_URI}"

echo "Step 1: Authenticate and configure Docker"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "Step 2: Build and push Docker image"
docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

echo "Step 3: Deploy to Cloud Run"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --port 8000

echo "Deployment complete! Retrieve the service URL with:"
gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)'
