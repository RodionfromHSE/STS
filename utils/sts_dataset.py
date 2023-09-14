from torch.utils.data import Dataset


class TextSimilarityDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids_1 = self.dataset[idx]["input_ids_1"]
        attention_mask_1 = self.dataset[idx]["attention_mask_1"]
        input_ids_2 = self.dataset[idx]["input_ids_2"]
        attention_mask_2 = self.dataset[idx]["attention_mask_2"]
        score = self.dataset[idx]["score"]
        return (input_ids_1, attention_mask_1, input_ids_2, attention_mask_2), score
