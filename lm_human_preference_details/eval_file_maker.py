from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM, GenerationConfig
from data import DATASET
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import json
from tqdm import tqdm

print("imports done")

query_len = 600
resp_len = 424
temp = 0.7
model_file = "sft"
model_name = "kkanarios/gpt2-tldr-sft"

indices = [3, 9, 10, 12, 18, 21, 28, 34, 36, 39, 44, 45, 55, 57, 58, 63, 64, 71, 78, 83, 84, 92, 95, 102, 103, 118, 149, 152, 166, 168, 178, 192, 193, 196, 202, 227, 233, 238, 240, 244, 257, 282, 287, 288, 290, 306, 312, 314, 316, 322, 323, 333, 335, 344, 350, 353, 365, 367, 369, 371, 372, 376, 386, 388, 393, 395, 406, 408, 419, 429, 436, 437, 445, 449, 454, 460, 468, 478, 492, 516, 531, 540, 544, 547, 551, 552, 555, 557, 559, 576, 587, 594, 595, 596, 598, 605, 611, 613, 617, 622, 625, 626, 631, 635, 642, 652, 653, 659, 662, 666, 668, 671, 672, 681, 684, 688, 699, 706, 712, 715, 717, 721, 725, 735, 739, 741, 747, 756, 766, 771, 775, 777, 779, 796, 797, 810, 830, 855, 856, 863, 868, 869, 874, 877, 881, 882, 887, 888, 894, 902, 916, 926, 928, 932, 934, 935, 937, 940, 941, 942, 946, 947, 954, 958, 959, 962, 971, 989, 993, 1016, 1021, 1040, 1047, 1049, 1063, 1064, 1084, 1104, 1105, 1111, 1113, 1117, 1122, 1128, 1132, 1151, 1155, 1164, 1172, 1175, 1179, 1185, 1205, 1219, 1261, 1265, 1271, 1273, 1287, 1298, 1301, 1308, 1314, 1330, 1332, 1341, 1345, 1352, 1364, 1370, 1371, 1393, 1406, 1407, 1409, 1410, 1416, 1417, 1418, 1424, 1435, 1440, 1441, 1446, 1450, 1452, 1454, 1463, 1477, 1485, 1505, 1507, 1510, 1517, 1521, 1531, 1540, 1542, 1548, 1552, 1558, 1564, 1566, 1572, 1576, 1599, 1605, 1607, 1612, 1614, 1625, 1634, 1637, 1645, 1646, 1657, 1658, 1672, 1673, 1684, 1689, 1692, 1705, 1707, 1714, 1718, 1723, 1731, 1741, 1744, 1758, 1759, 1764, 1765, 1776, 1778, 1793, 1794, 1801, 1817, 1825, 1836, 1842, 1844, 1846, 1847, 1854, 1858, 1861, 1870, 1871, 1872, 1873, 1877, 1884, 1894, 1895, 1898, 1922, 1928, 1932, 1937, 1938, 1939, 1940, 1945, 1946, 1949, 1950, 1959, 1963, 1969, 1985, 1988, 1998, 2015, 2018, 2040, 2046, 2051, 2052, 2055, 2077, 2080, 2084, 2087, 2088, 2091, 2101, 2103, 2107, 2111, 2113, 2115, 2121, 2145, 2150, 2152, 2157, 2158, 2160, 2166, 2175, 2179, 2181, 2191, 2192, 2212, 2215, 2218, 2222, 2232, 2260, 2261, 2276, 2278, 2279, 2283, 2285, 2286, 2292, 2298, 2299, 2323, 2335, 2339, 2340, 2342, 2351, 2362, 2369, 2383, 2386, 2390, 2392, 2395, 2398, 2400, 2427, 2429, 2433, 2440, 2446, 2451, 2460, 2467, 2469, 2471, 2474, 2475, 2481, 2484, 2488, 2495, 2497, 2501, 2504, 2505, 2509, 2510, 2528, 2532, 2541, 2554, 2555, 2573, 2574, 2591, 2603, 2604, 2611, 2613, 2614, 2618, 2619, 2651, 2654, 2666, 2674, 2693, 2701, 2707, 2720, 2722, 2739, 2741, 2750, 2755, 2760, 2763, 2765, 2774, 2790, 2793, 2802, 2810, 2821, 2823, 2834, 2839, 2840, 2841, 2844, 2858, 2881, 2883, 2903, 2909, 2912, 2921, 2933, 2935, 2936, 2939, 2944, 2945, 2948, 2954, 2965, 2966, 2969, 2970, 2974, 2985, 2988, 2991, 3003, 3005, 3006, 3010, 3012, 3032, 3044, 3057, 3061, 3063, 3064, 3067, 3070, 3077, 3094, 3100, 3103, 3108, 3115, 3118, 3119, 3130, 3137, 3139, 3140, 3150, 3156, 3158, 3159, 3165, 3169, 3174, 3180, 3185, 3195, 3203, 3237, 3240, 3243, 3246, 3247, 3248, 3249, 3253, 3254, 3255, 3264, 3278, 3286, 3288, 3306, 3329, 3348, 3351, 3373, 3390, 3397, 3407, 3408, 3409, 3415, 3416, 3424, 3442, 3444, 3447, 3453, 3458, 3460, 3474, 3480, 3488, 3507, 3513, 3514, 3522, 3523, 3536, 3545, 3551, 3552, 3558, 3560, 3561, 3563, 3570, 3574, 3586, 3587, 3603, 3625, 3629, 3631, 3639, 3647, 3649, 3650, 3656, 3660, 3667, 3668, 3670, 3680, 3682, 3697, 3707, 3718, 3721, 3730, 3731, 3743, 3748, 3750, 3751, 3752, 3761, 3763, 3767, 3776, 3789, 3790, 3794, 3796, 3811, 3817, 3821, 3825, 3834, 3838, 3844, 3846, 3851, 3853, 3859, 3862, 3864, 3865, 3874, 3877, 3886, 3887, 3899, 3900, 3904, 3905, 3906, 3908, 3910, 3927, 3935, 3938, 3965, 3976, 3979, 3981, 3982, 3988, 3989, 3991, 3992, 4000, 4009, 4014, 4022, 4031, 4047, 4051, 4058, 4062, 4063, 4064, 4067, 4072, 4087, 4089, 4091, 4098, 4103, 4115, 4119, 4124, 4147, 4156, 4169, 4181, 4184, 4188, 4193, 4196, 4205, 4207, 4208, 4210, 4211, 4212, 4217, 4226, 4228, 4229, 4237, 4245, 4250, 4256, 4257, 4260, 4262, 4266, 4274, 4279, 4282, 4287, 4290, 4293, 4294, 4299, 4302, 4307, 4311, 4312, 4315, 4320, 4348, 4365, 4368, 4383, 4387, 4393, 4395, 4400, 4402, 4417, 4423, 4431, 4436, 4437, 4440, 4441, 4443, 4452, 4468, 4476, 4482, 4487, 4498, 4502, 4507, 4511, 4524, 4531, 4532, 4533, 4534, 4537, 4544, 4548, 4558, 4561, 4562, 4573, 4576, 4581, 4582, 4593, 4599, 4604, 4605, 4610, 4617, 4623, 4629, 4630, 4634, 4652, 4677, 4679, 4681, 4691, 4698, 4721, 4727, 4734, 4736, 4741, 4751, 4762, 4773, 4781, 4785, 4788, 4791, 4793, 4797, 4798, 4799, 4812, 4821, 4826, 4840, 4861, 4864, 4868, 4869, 4880, 4883, 4885, 4890, 4896, 4900, 4901, 4902, 4907, 4910, 4918, 4919, 4922, 4936, 4945, 4948, 4953, 4954, 4974, 4975, 4979, 4984, 4993, 4997, 5003, 5007, 5017, 5027, 5040, 5055, 5065, 5070, 5071, 5077, 5089, 5116, 5125, 5131, 5141, 5145, 5147, 5151, 5164, 5170, 5172, 5174, 5196, 5199, 5208, 5210, 5216, 5222, 5239, 5244, 5245, 5259, 5261, 5263, 5266, 5270, 5272, 5273, 5282, 5292, 5294, 5307, 5313, 5317, 5318, 5323, 5326, 5328, 5342, 5346, 5359, 5374, 5377, 5379, 5398, 5400, 5401, 5404, 5407, 5414, 5422, 5433, 5435, 5443, 5449, 5468, 5473, 5474, 5475, 5480, 5496, 5506, 5511, 5512, 5541, 5543, 5553, 5554, 5559, 5565, 5579, 5584, 5594, 5604, 5609, 5621, 5622, 5641, 5647, 5653, 5667, 5669, 5670, 5677, 5694, 5702, 5707, 5709, 5710, 5715, 5724, 5728, 5732, 5733, 5756, 5762, 5775, 5777, 5785, 5802, 5803, 5809, 5816, 5819, 5824, 5825, 5827, 5842, 5850, 5854, 5855, 5857, 5862, 5868, 5870, 5871, 5885, 5900, 5903, 5911, 5918, 5955, 5959, 5985, 5994, 5996, 6014, 6016, 6021, 6027, 6032, 6042, 6048, 6055, 6065, 6075, 6096, 6107, 6108, 6130, 6131, 6139, 6141, 6146, 6154, 6163, 6165, 6169, 6177, 6183, 6184, 6188, 6191, 6195, 6196, 6208, 6220, 6224, 6230, 6231, 6234, 6235, 6236, 6244, 6245, 6248, 6255, 6258, 6281, 6285, 6291, 6306, 6308, 6309, 6313, 6314, 6317, 6320, 6326, 6336, 6346, 6362, 6364, 6378, 6399, 6403, 6414, 6419, 6425, 6427, 6433, 6440, 6442, 6443, 6455, 6463, 6478, 6485]

class MySFTDataset(IterableDataset):
    def __init__(self, generator, tokenizer, query_length, seed, start_text=None, end_text=None):
        self.generator = generator
        self.tokenizer = tokenizer
        self.query_length = query_length
        self.start_text = start_text
        self.end_text = end_text
        self.seed = seed
        token_to_index = tokenizer.get_vocab()
        self.start_token = token_to_index[start_text] if self.start_text else None
        self.end_token = token_to_index[end_text] if self.end_text else None

    def __iter__(self):
        for query, response in self.generator("test", self.seed, shuffle=True):
            query_tokens = self.tokenizer.encode(query)

            if self.start_token is not None:
                try:
                    first_index = query_tokens.index(self.start_token) + 1
                    if first_index < len(query_tokens):
                        query_tokens = query_tokens[first_index:]
                except:
                    continue

            query_tokens = query_tokens[: self.query_length]
            if self.end_token is not None:
                try:
                    last_index = len(query_tokens) - query_tokens[::-1].index(self.end_token)
                    query_tokens = query_tokens[:last_index]
                except:
                    continue

            query_output = self.tokenizer.pad(
                {"input_ids": query_tokens},
                padding="max_length",
                max_length=self.query_length,
                return_tensors="np",
                return_attention_mask=False,
            )

            max_response_length = resp_len
            response_tokens = self.tokenizer.encode(response, max_length=max_response_length,
                                                 truncation=True)
            response_output = self.tokenizer.pad({"input_ids": response_tokens},
                                                 padding="max_length",
                                                 max_length=max_response_length,
                                                 return_tensors = "np",
                                                 return_attention_mask=False
                                                )

            yield query_output["input_ids"], np.squeeze(response_output["input_ids"]), query, response
            
print("read the function")
tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        padding_side="right",
    )
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

print("tokenizer initialized")

dataset = MySFTDataset(
        DATASET["tldr-sft"],
        tokenizer,
        query_len,
        seed=10,
        start_text=None,
        end_text=None,
    )

print("dataset generated")
dataset = iter(dataset)

lm_backbone = FlaxAutoModelForCausalLM.from_pretrained(model_name)
lm_backbone.generation_config.eos_token_id = None
lm_backbone.generation_config.pad_token_id = tokenizer.pad_token_id

generation_config = GenerationConfig(
    max_new_tokens=resp_len,
    temperature=temp,
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
)

def policy_generate(
        model_params,
        queries: jnp.ndarray,
    ):
        input_ids = queries
        attention_mask = input_ids != tokenizer.pad_token_id
        input_ids = jnp.where(attention_mask, queries, 0)
        output = lm_backbone.generate(
            params=model_params,
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=attention_mask.astype("i4"),
            return_dict_in_generate=True,
        )
        query_length = input_ids.shape[1]
        return jnp.concatenate((queries, output.sequences[:, query_length:]), axis=1)
    
print("lm backbone created")

with open(model_file+".jsonl", "w") as outfile:
    for idx, elem in tqdm(enumerate(dataset)):
        if idx in indices:
            query, response, query_words, response_words = elem
            
            gen_response = policy_generate(lm_backbone.params, np.reshape(query, (1, query_len)))
            model_response = tokenizer.decode(gen_response[0, query_len:])
            entry = {"query": query_words, "response": model_response}
            
            print(json.dumps(entry), file=outfile)