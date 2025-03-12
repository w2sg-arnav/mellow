import torch
from pathlib import Path
import os
from mellow import MellowWrapper
    
if __name__ == "__main__":
    # setup cuda and device
    cuda = torch.cuda.is_available()
    device = 0 if cuda else "cpu"

    # setup mellow
    mellow = MellowWrapper(
                        config = "v0",
                        model = "v0",
                        device=device,
                        use_cuda=cuda,
                    )
    
    # pick up audio file paths
    parent_path = Path(os.path.realpath(__file__)).parent
    path1 = os.path.join(parent_path, "resource", "1.wav")
    path2 = os.path.join(parent_path, "resource", "2.wav")
    
    # list of filepaths and prompts
    examples = [
        [path1, path2, "what is the primary sound event present in the clip? a) dog barking b) chirping birds c) car engine d) clapping"], # defaults to first audio
    ]

    # generate response
    response = mellow.generate(examples=examples, max_len=300, top_p=0.8, temperature=1.0)
    print(f"\noutput: {response}")