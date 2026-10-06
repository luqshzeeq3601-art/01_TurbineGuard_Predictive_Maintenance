# PowerShell deployment helper script for TurbineGuard on GCP Cloud Run
param (
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = "asia-southeast1",
    [string]$ServiceName = "turbineguard-api",
    [string]$RepoName = "turbineguard-repo",
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    Write-Error "Please specify -ProjectId or set GCP_PROJECT_ID environment variable."
}

$ImageUri = "${Region}-docker.pkg.dev/${ProjectId}/${RepoName}/${ServiceName}:${ImageTag}"

Write-Host "=== TurbineGuard Cloud Run Deployment ===" -ForegroundColor Cyan
Write-Host "Project ID: $ProjectId"
Write-Host "Region:     $Region"
Write-Host "Image URI:  $ImageUri"

Write-Host "`nStep 1: Authenticate and configure Docker..." -ForegroundColor Yellow
gcloud auth configure-docker "${Region}-docker.pkg.dev" --quiet

Write-Host "`nStep 2: Build and push Docker image..." -ForegroundColor Yellow
docker build -t $ImageUri .
docker push $ImageUri

Write-Host "`nStep 3: Deploy to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image $ImageUri `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --memory 512Mi `
    --cpu 1 `
    --min-instances 0 `
    --max-instances 2 `
    --port 8000

Write-Host "`nDeployment complete! Service URL:" -ForegroundColor Green
gcloud run services describe $ServiceName --platform managed --region $Region --format 'value(status.url)'
