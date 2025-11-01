pipeline {
    agent any
    
    environment {
        // Azure credentials from Jenkins
        AZURE_CLIENT_ID = credentials('azure-client-id')
        AZURE_CLIENT_SECRET = credentials('azure-client-secret')
        AZURE_TENANT_ID = credentials('azure-tenant-id')
        AZURE_SUBSCRIPTION_ID = credentials('azure-subscription-id')
        
        // ACR credentials
        ACR_USERNAME = credentials('acr-username')
        ACR_PASSWORD = credentials('acr-password')
        
        // Storage credentials
        STORAGE_ACCOUNT_NAME = credentials('storage-account-name')
        STORAGE_ACCOUNT_KEY = credentials('storage-account-key')
        
        // API Keys (add these to Jenkins)
        OPENAI_API_KEY = credentials('openai-api-key')
        GOOGLE_API_KEY = credentials('google-api-key')
        GROQ_API_KEY = credentials('groq-api-key')
        TAVILY_API_KEY = credentials('tavily-api-key')
        LLM_PROVIDER = credentials('llm-provider')
        
        // App configuration
        APP_RESOURCE_GROUP = 'research-report-app-rg'
        APP_NAME = 'research-report-app'
        ACR_NAME = 'researchreportacr'
        IMAGE_NAME = 'research-report-app'
        CONTAINER_ENV = 'research-report-env'
    }
    
    stages {
        stage('Checkout') {
    steps {
        echo 'Checking out code from Git...'
        // Clean workspace first
        deleteDir()
        // Clone the repository
        git branch: 'main',
            url: 'https://github.com/uchihamadara91/AutomatedResearchReportGeneration.git'
    }
}
        
        stage('Setup Python Environment') {
            steps {
                echo 'Setting up Python environment...'
                sh '''
                    python3 --version
                    # Use --break-system-packages since we're in a container
                    python3 -m pip install --upgrade pip --break-system-packages
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                sh '''
                    # Install in user space to avoid system-wide issues
                    pip3 install -r requirements.txt --break-system-packages
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                echo 'Running tests...'
                sh '''
                    # Add your test commands here
                    # For now, just verify imports work
                    python3 -c "from research_and_analyst.api.main import app; print('✅ Imports successful')"
                '''
            }
        }
        
        stage('Login to Azure') {
            steps {
                echo 'Logging in to Azure...'
                sh '''
                    az login --service-principal \
                      -u $AZURE_CLIENT_ID \
                      -p $AZURE_CLIENT_SECRET \
                      --tenant $AZURE_TENANT_ID
                    
                    az account set --subscription $AZURE_SUBSCRIPTION_ID
                '''
            }
        }
        
        stage('Verify Docker Image in ACR') {
            steps {
                echo 'Verifying Docker image exists in ACR...'
                script {
                    // Get the latest tag from ACR
                    def imageTag = sh(
                        script: """
                            az acr repository show-tags \
                              --name \$ACR_NAME \
                              --repository \$IMAGE_NAME \
                              --orderby time_desc \
                              --output tsv \
                              | head -n 1
                        """,
                        returnStdout: true
                    ).trim()
                    
                    if (imageTag) {
                        echo "Found image with tag: ${imageTag}"
                        env.IMAGE_TAG = imageTag
                    } else {
                        error "No images found in ACR. Please run: ./build-and-push-docker-image.sh"
                    }
                    
                    // Show all available tags
                    sh """
                        echo "📋 Available tags:"
                        az acr repository show-tags \
                          --name \$ACR_NAME \
                          --repository \$IMAGE_NAME \
                          --output table
                    """
                }
            }
        }
        
        stage('Deploy to Azure Container Apps') {
            steps {
                echo 'Deploying to Azure Container Apps...'
                sh '''
                    echo "Using image: ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"
                    
                    # Check if Container App exists
                    if az containerapp show \
                      --name $APP_NAME \
                      --resource-group $APP_RESOURCE_GROUP > /dev/null 2>&1; then
                        echo "Checking current Container App state..."
                        CURRENT_STATE=$(az containerapp show \
                          --name $APP_NAME \
                          --resource-group $APP_RESOURCE_GROUP \
                          --query properties.provisioningState -o tsv)
                        echo "Current state: $CURRENT_STATE"
                        
                        if [ "$CURRENT_STATE" != "Succeeded" ]; then
                          echo "App state is '$CURRENT_STATE'. Recreating the Container App..."
                          az containerapp delete --name $APP_NAME --resource-group $APP_RESOURCE_GROUP --yes
                          
                          echo "🆕 Creating new Container App..."
                          az containerapp create \
                            --name $APP_NAME \
                            --resource-group $APP_RESOURCE_GROUP \
                            --environment $CONTAINER_ENV \
                            --image ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG} \
                            --registry-server ${ACR_NAME}.azurecr.io \
                            --registry-username $ACR_USERNAME \
                            --registry-password $ACR_PASSWORD \
                            --target-port 8000 \
                            --ingress external \
                            --min-replicas 1 \
                            --max-replicas 3 \
                            --cpu 1.0 \
                            --memory 2.0Gi \
                            --env-vars LLM_PROVIDER=$LLM_PROVIDER
                          
                          # Wait for Succeeded state
                          echo "Waiting for Container App to reach Succeeded state..."
                          for i in $(seq 1 30); do
                            STATE=$(az containerapp show --name $APP_NAME --resource-group $APP_RESOURCE_GROUP --query properties.provisioningState -o tsv)
                            echo "Attempt $i: state=$STATE"
                            if [ "$STATE" = "Succeeded" ]; then
                              break
                            fi
                            sleep 10
                          done
                          
                          # Add secrets
                          echo "Adding secrets..."
                          az containerapp secret set \
                            --name $APP_NAME \
                            --resource-group $APP_RESOURCE_GROUP \
                            --secrets \
                              openai-api-key=$OPENAI_API_KEY \
                              google-api-key=$GOOGLE_API_KEY \
                              groq-api-key=$GROQ_API_KEY \
                              tavily-api-key=$TAVILY_API_KEY
                          
                          # Link env vars to secrets
                          echo "Linking environment variables to secrets..."
                          az containerapp update \
                            --name $APP_NAME \
                            --resource-group $APP_RESOURCE_GROUP \
                            --set-env-vars \
                              OPENAI_API_KEY=secretref:openai-api-key \
                              GOOGLE_API_KEY=secretref:google-api-key \
                              GROQ_API_KEY=secretref:groq-api-key \
                              TAVILY_API_KEY=secretref:tavily-api-key \
                              LLM_PROVIDER=$LLM_PROVIDER
                        else
                          echo "Updating existing Container App..."
                          
                          # Update secrets first
                          echo "Updating secrets..."
                          az containerapp secret set \
                            --name $APP_NAME \
                            --resource-group $APP_RESOURCE_GROUP \
                            --secrets \
                              openai-api-key=$OPENAI_API_KEY \
                              google-api-key=$GOOGLE_API_KEY \
                              groq-api-key=$GROQ_API_KEY \
                              tavily-api-key=$TAVILY_API_KEY
                          
                          # Update container app with new image
                          az containerapp update \
                            --name $APP_NAME \
                            --resource-group $APP_RESOURCE_GROUP \
                            --image ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}
                        fi
                    else
                        echo "Creating new Container App..."
                        
                        # Step 1: Create the app without secret references
                        az containerapp create \
                          --name $APP_NAME \
                          --resource-group $APP_RESOURCE_GROUP \
                          --environment $CONTAINER_ENV \
                          --image ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG} \
                          --registry-server ${ACR_NAME}.azurecr.io \
                          --registry-username $ACR_USERNAME \
                          --registry-password $ACR_PASSWORD \
                          --target-port 8000 \
                          --ingress external \
                          --min-replicas 1 \
                          --max-replicas 3 \
                          --cpu 1.0 \
                          --memory 2.0Gi \
                          --env-vars LLM_PROVIDER=$LLM_PROVIDER
                        
                        # Wait for Succeeded state before setting secrets
                        echo "Waiting for Container App to reach Succeeded state..."
                        for i in $(seq 1 30); do
                          STATE=$(az containerapp show --name $APP_NAME --resource-group $APP_RESOURCE_GROUP --query properties.provisioningState -o tsv)
                          echo "Attempt $i: state=$STATE"
                          if [ "$STATE" = "Succeeded" ]; then
                            break
                          fi
                          sleep 10
                        done
                        
                        # Step 2: Add secrets
                        echo "Adding secrets..."
                        az containerapp secret set \
                          --name $APP_NAME \
                          --resource-group $APP_RESOURCE_GROUP \
                          --secrets \
                            openai-api-key=$OPENAI_API_KEY \
                            google-api-key=$GOOGLE_API_KEY \
                            groq-api-key=$GROQ_API_KEY \
                            tavily-api-key=$TAVILY_API_KEY
                        
                        # Step 3: Update env vars to use secret references
                        echo "Linking environment variables to secrets..."
                        az containerapp update \
                          --name $APP_NAME \
                          --resource-group $APP_RESOURCE_GROUP \
                          --set-env-vars \
                            OPENAI_API_KEY=secretref:openai-api-key \
                            GOOGLE_API_KEY=secretref:google-api-key \
                            GROQ_API_KEY=secretref:groq-api-key \
                            TAVILY_API_KEY=secretref:tavily-api-key \
                            LLM_PROVIDER=$LLM_PROVIDER
                    fi
                '''
            }
        }
        
        stage('Verify Deployment') {
            steps {
                echo 'Verifying deployment...'
                sh '''
                    # Get app URL
                    APP_URL=$(az containerapp show \
                      --name $APP_NAME \
                      --resource-group $APP_RESOURCE_GROUP \
                      --query properties.configuration.ingress.fqdn -o tsv)
                    
                    echo "Application URL: https://$APP_URL"
                    
                    # Wait for app to be ready
                    echo "Waiting for application to be ready..."
                    sleep 30
                    
                    # Health check
                    if curl -f -s https://$APP_URL/health > /dev/null; then
                        echo "Application is responding!"
                    else
                        echo "Application may still be starting up..."
                    fi
                '''
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
        always {
            echo 'Cleaning up workspace...'
            cleanWs()
        }
    }
}
