# How to Print Documentation to PDF

## Files Created

I've created the following documentation files for you:

### 1. Model Architecture Explanation
- **Markdown**: `TRM_Model_Architecture_Explained.md`
- **HTML** (for printing): `TRM_Model_Architecture_Explained.html`
- **Content**: Complete explanation of how your TRM model architecture works

### 2. Training Pipeline Explanation
- **Markdown**: `TRM_Training_Pipeline_Explained.md`
- **HTML** (for printing): `TRM_Training_Pipeline_Explained.html`
- **Content**: Complete explanation of data generation, training, and evaluation

## How to Convert HTML to PDF

### Option 1: Using Your Web Browser (Easiest)

1. **Open the HTML file** in any web browser (Chrome, Firefox, Safari, Edge):
   ```bash
   # From your local machine, navigate to:
   docs/TRM_Model_Architecture_Explained.html
   docs/TRM_Training_Pipeline_Explained.html
   ```

2. **Print to PDF**:
   - Press `Ctrl+P` (Windows/Linux) or `Cmd+P` (Mac)
   - In the print dialog, select **"Save as PDF"** or **"Microsoft Print to PDF"**
   - Recommended settings:
     - Paper size: **A4** or **Letter**
     - Margins: **Default** or **Minimum**
     - Scale: **100%** or **Fit to page**
     - Background graphics: **ON** (for code highlighting)
   - Click **"Save"**

3. **Done!** You now have a PDF file.

### Option 2: Using Command Line (If wkhtmltopdf is available)

```bash
# Install wkhtmltopdf (if not available)
# Ubuntu/Debian:
sudo apt-get install wkhtmltopdf

# Mac:
brew install wkhtmltopdf

# Convert to PDF:
wkhtmltopdf docs/TRM_Model_Architecture_Explained.html docs/TRM_Model_Architecture_Explained.pdf
wkhtmltopdf docs/TRM_Training_Pipeline_Explained.html docs/TRM_Training_Pipeline_Explained.pdf
```

### Option 3: Using Online Converter

1. Go to any online HTML to PDF converter:
   - https://www.html2pdf.com/
   - https://cloudconvert.com/html-to-pdf
   - https://www.sejda.com/html-to-pdf

2. Upload the HTML file
3. Download the PDF

## File Locations

All files are in:
```
/orcd/home/002/amitjain/project/TinyRecursiveControl/docs/
├── TRM_Model_Architecture_Explained.md      (11 KB)
├── TRM_Model_Architecture_Explained.html    (15 KB)
├── TRM_Training_Pipeline_Explained.md       (16 KB)
└── TRM_Training_Pipeline_Explained.html     (20 KB)
```

## Print Quality Tips

For best print quality:
- **Use Chrome or Edge** - they have the best print rendering
- **Enable background graphics** - ensures code blocks are highlighted
- **Use A4 or Letter paper size** - documents are formatted for these sizes
- **Check margins** - default margins work well, but you can adjust if needed

## What's Inside

### Document 1: Model Architecture
- What is TRM?
- Your TRC adaptation
- Two architectural modes (single-latent vs two-level)
- Step-by-step execution walkthrough
- Key mechanisms (recursive reasoning, weight sharing, context injection)
- Ablation study results
- Visual explanations

### Document 2: Training Pipeline
- Dataset generation (optimal teacher)
- Training process (supervised learning)
- How recursive reasoning helps learning
- Evaluation methodology
- Complete data flow diagrams
- Why it's NOT reinforcement learning
- Code references

## Questions?

If you have any questions about the content or need clarification on any topic, feel free to ask!
