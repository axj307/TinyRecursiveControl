#!/usr/bin/env node

const PptxGenJS = require('pptxgenjs');
const html2pptx = require('./scripts/html2pptx.js');
const fs = require('fs');
const path = require('path');

async function createPresentation() {
  console.log('Creating TRC presentation...');

  // Create presentation with 16:9 layout
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'TinyRecursiveControl';
  pptx.title = 'Tiny Recursive Control: Parameter-Efficient Control Synthesis';
  pptx.subject = 'Recursive Reasoning for Optimal Control';

  // Slide files in order
  const slideFiles = [
    'slide1.html',
    'slide2.html',
    'slide3.html',
    'slide3_trm_style.html',
    'slide3_5.html',
    'slide4.html',
    'slide5.html'
  ];

  console.log('Converting HTML slides to PowerPoint...');

  for (let i = 0; i < slideFiles.length; i++) {
    const slideFile = slideFiles[i];
    const slidePath = path.join(__dirname, slideFile);

    if (!fs.existsSync(slidePath)) {
      console.error(`Error: Slide file not found: ${slideFile}`);
      continue;
    }

    console.log(`  Processing ${slideFile}...`);

    try {
      await html2pptx(slidePath, pptx);
      console.log(`    ✓ ${slideFile} converted successfully`);
    } catch (error) {
      console.error(`    ✗ Error converting ${slideFile}:`, error.message);
      throw error;
    }
  }

  // Save the presentation
  const outputPath = path.join(__dirname, 'TRC_Presentation.pptx');
  console.log(`\nSaving presentation to: ${outputPath}`);

  await pptx.writeFile({ fileName: outputPath });

  console.log('✓ Presentation created successfully!');
  console.log(`\nOutput: ${outputPath}`);

  return outputPath;
}

// Run the function
createPresentation()
  .then((outputPath) => {
    console.log('\n========================================');
    console.log('SUCCESS!');
    console.log('========================================');
    console.log(`Presentation saved to: ${outputPath}`);
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n========================================');
    console.error('ERROR!');
    console.error('========================================');
    console.error(error);
    process.exit(1);
  });
