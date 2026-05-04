from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
from dataloaders.rawvideo_util import RawVideoExtractor
from dataloaders.rawframes_util import RawFrameExtractor

class DiDeMo_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            video_data_type='video',
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.video_data_type = video_data_type
        assert self.video_data_type in ['video', 'frames']

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_path, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")

        video_json_path_dict = {}
        video_json_path_dict["train"] = os.path.join(self.data_path, "train_data.json")
        video_json_path_dict["val"] = os.path.join(self.data_path, "val_data.json")
        video_json_path_dict["test"] = os.path.join(self.data_path, "test_data.json")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
        for itm in json_data:
            description = itm["description"]
            times = itm["times"]
            video = itm["video"]
            if video not in video_ids:
                continue

            # each video is split into 5-second temporal chunks
            # average the points from each annotator
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        for k_ in caption_dict.keys():
            caption_dict[k_]["start"] = [0]
            # trick to save time on obtaining each video length
            # [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
            # Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
            caption_dict[k_]["end"] = [31]
            caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

        video_dict = {}
        if self.video_data_type == 'frames':
            # For frames: expect directories containing frame images
            for item_name in os.listdir(self.features_path):
                item_path = os.path.join(self.features_path, item_name)
                if not os.path.isdir(item_path):
                    continue
                if item_name not in video_ids:
                    continue
                video_dict[item_name] = item_path
        else:
            # For videos: expect video files
            for root, dub_dir, video_files in os.walk(self.features_path):
                for video_file in video_files:
                    video_id_ = video_file
                    if video_id_ not in video_ids:
                        continue
                    file_path_ = os.path.join(root, video_file)
                    video_dict[video_id_] = file_path_

        self.caption_dict = caption_dict
        self.video_dict = video_dict
        video_ids = list(set(video_ids) & set(self.caption_dict.keys()) & set(self.video_dict.keys()))

        # Get all captions
        self.iter2video_pairs_dict = {}
        for video_id in self.caption_dict.keys():
            if video_id not in video_ids:
                continue
            caption = self.caption_dict[video_id]
            n_caption = len(caption['start'])
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (video_id, sub_id)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFrameExtractor = RawFrameExtractor(size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        caption = self.caption_dict[video_id]
        k = 1
        r_ind = [sub_id]

        starts = np.zeros(k, dtype=np.long)
        ends = np.zeros(k, dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            start_, end_ = caption['start'][ind], caption['end'][ind]
            words = self.tokenizer.tokenize(caption['text'][ind])
            starts[i], ends[i] = start_, end_

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, starts, ends

    def _get_rawvideo(self, idx, s, e):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)
        video_path = self.video_dict[idx]

        try:
            for i in range(len(s)):
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = start_time if start_time >= 0. else 0.
                end_time = end_time if end_time >= 0. else 0.
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                cache_id = "{}_{}_{}".format(video_path, start_time, end_time)
                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
                raw_video_data = raw_video_data['video']

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e, excep))
            pass
            # raise e

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawframes(self, idx):
        """Get frames from directory (frames mode). idx is video_id."""
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        max_video_length = [0]

        # Pair x L x T x 3 x H x W
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.rawFrameExtractor.size, self.rawFrameExtractor.size), dtype=float)

        try:
            frames_path = self.video_dict[idx]
            
            if not os.path.isdir(frames_path):
                print(f"Frames directory not found: {frames_path}")
                return video, video_mask

            frames_data = self.rawFrameExtractor.get_frames_data(frames_path)
            if frames_data is None:
                print(f"Failed to get frames from: {frames_path}")
                return video, video_mask

            raw_frames_data = frames_data['frames']  # L x 3 x H x W (torch tensor)

            if len(raw_frames_data.shape) > 3:
                # L x 3 x H x W -> L x 1 x 3 x H x W (add temporal dimension)
                raw_frames_data_clip = raw_frames_data.unsqueeze(1)  # L x 1 x 3 x H x W
                raw_frames_data_clip = raw_frames_data_clip.numpy()  # Convert to numpy

                # Apply frame sampling based on max_frames
                if self.max_frames < raw_frames_data_clip.shape[0]:
                    if self.slice_framepos == 0:
                        frame_slice = raw_frames_data_clip[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        frame_slice = raw_frames_data_clip[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_frames_data_clip.shape[0] - 1, num=self.max_frames, dtype=int)
                        frame_slice = raw_frames_data_clip[sample_indx, ...]
                else:
                    frame_slice = raw_frames_data_clip

                # Apply frame order
                if self.frame_order == 1:
                    frame_slice = frame_slice[::-1, ...]
                elif self.frame_order == 2:
                    np.random.shuffle(frame_slice)

                slice_len = frame_slice.shape[0]
                max_video_length[0] = slice_len
                if slice_len > 0:
                    video[0][:slice_len, ...] = frame_slice
        except Exception as excep:
            print(f"Error loading frames for video id: {idx}, Error: {excep}")
            pass

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, feature_idx):
        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

        pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(video_id, sub_id)
        if self.video_data_type == 'video':
            # Video mode: use temporal slicing based on start/end times
            video, video_mask = self._get_rawvideo(video_id, starts, ends)
        else:  # 'frames'
            # Frames mode: load all frames from directory, ignore start/end times
            video, video_mask = self._get_rawframes(video_id)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
