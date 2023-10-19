## Evaluation datasets

1. Please download the data from this [repository](https://huggingface.co/BAAI/Uni3D/blob/main/data/test_datasets.zip), which contains datasets for Objaverse-LVIS, ModelNet40, and ScanObjectNN.

2. Place the `test_datasets` folder in the `/data` directory on your machine. The core `data` directory structure should look like this:

   ```
   ./data 
   -- test_datasets/  
      -- modelnet40
      -- scanobjectnn
      -- objaverse_lvis
   -- utils/
   -- datasets.py
   -- ModelNet40_openshape.yaml
   -- Objaverse_lvis_openshape.yaml
   -- ScanObjNN_openshape.yaml
   -- dataset_catalog.json
   -- labels.json
   -- templates.json
   ```
3. **Important**: If you choose to place the data in a location other than the default one mentioned above, please remember to update the corresponding dataset's YAML file with your path.

Now you are ready to use the datasets for zero-shot evaluation. If you have any questions or encounter any issues, please refer to the documentation or feel free to reach out for assistance.

## Pre-training datasets

We're in the process of organizing and uploading. Hang tight, and stay tuned! ☕️

Thanks for your patience and support!
