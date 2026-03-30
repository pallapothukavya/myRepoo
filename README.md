---
title: WBC Classification
emoji: 🩸
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# WBC Classification Using Django and CNN

This project classifies White Blood Cells (WBC) into different categories using a Convolutional Neural Network (CNN).

## Deployment on Hugging Face Spaces
This project is configured to run on Hugging Face Spaces using the **Docker SDK**.

### Features:
- **Mobile Friendly**: Optimized file upload for mobile browsers.
- **Responsive Layout**: Sidebar collapses on small screens.
- **Static Hosting**: Uses Whitenoise for efficient static file serving.

### Manual Steps to Deploy:
1. Create a **New Space** on Hugging Face.
2. Set the name and select **Docker** as the SDK.
3. Choose **Import from GitHub** and paste your repository URL.
4. Hugging Face will automatically build and deploy based on the `Dockerfile`.
