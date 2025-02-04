from theorder import LoBertModel, LoBertConfig
from datasets import Dataset

if __name__ == '__main__':
    config = LoBertConfig()
    model = LoBertModel(config)

    # path_data_sample = "/workspaces/2025 LoBERT/data/LOBSTER_SampleFile_AAPL_2012-06-21_10/ArrowDataset"
    # ds = Dataset.load_from_disk(path_data_sample)