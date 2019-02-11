import numpy as np

class TokenizePages(object):
    def __init__(self, page_clicks):
        lookups, longest_session = self.create_lookups_from_clicks(page_clicks)
        self.tokenize_representation = self.process_input_data(page_clicks, lookups, longest_session)

    def get_unique_urls(self,clicks):
        unique = []
        longest_session = 0
        for p_list in clicks:
            for i in range(0, len(p_list)):
                if len(p_list) > longest_session:
                    longest_session = len(p_list)
                url = p_list[i].split("https://")[-1]
                if url not in unique:
                    unique.append(url)
        return unique, longest_session

    def split_unique(self,unique_clicks):
        splitted_unique = []
        longest_url = 0
        for url in unique_clicks:
            splitted_url = url.split("/")
            splitted_unique.append(splitted_url)
            if len(splitted_url) > longest_url:
                longest_url = len(splitted_url)
        return splitted_unique, longest_url

    def get_token_lists(self, unique_splitted, longest_seq):
        # populate dictionary with unique url tokens
        d_out = {}
        for i in range(longest_seq):
            d_out[i] = []
        token_list = []
        for j in range(len(unique_splitted)): 
            for i in range(longest_seq):
                try:
                    if unique_splitted[j][i] not in d_out[i]:
                        d_out[i].append(unique_splitted[j][i])
                except IndexError:
                    continue
        return d_out

    def create_lookups(self, d, max_seq):
        d_out = {}
        prev_end_idx = 1
        for i in range(max_seq):
            urls = d[i]
            indices = list(range(prev_end_idx, prev_end_idx + len(urls)))
            prev_end_idx += len(urls)
            d_new = dict(zip(urls, indices))
            d_out[i] = d_new
        return d_out
    
    def create_lookups_from_clicks(self, clicks):
        unique_urls, longest_session = self.get_unique_urls(clicks)
        unique_splitted, max_seq = self.split_unique(unique_urls)
        token_dict = self.get_token_lists(unique_splitted, max_seq)
        return self.create_lookups(token_dict, max_seq), longest_session
    
    def process_url(self, url, lookup):
        out = []
        split_url = url.split("https://")[-1].split("/")
        splt_url_length = len(split_url)
        for i in range(len(split_url)):
            idx = lookup[i][split_url[i]]
            out.append(idx) + 1
        out.append(1)
        return out

    def process_session(self, session, lookup, max_sequence):
        for i, url in enumerate(session):
            if i == 0:
                out_arr = self.process_url(url, lookup)
            else:
                out_arr = np.append(out_arr, self.process_url(url, lookup))
        return out_arr[:-1]

    def process_input_data(self, data, lookup, largest_session):
        out_arr = []
        longest_seq = 0
        for i,page_session in enumerate(data):
            if len(page_session) >= 60:
                page_session = page_session[:59]
            out = self.process_session(page_session, lookup, largest_session)
            if len(out) > longest_seq:
                longest_seq = len(out)
            out_arr.append(out)
        out_padded = []
        for i in range(len(out_arr)):
            diff = longest_seq -  len(out_arr[i])
            padding = np.full(diff, -1)
            out_padded.append(np.append(out_arr[i],padding))
        return out_padded
    
    def run(self):
        return self.tokenize_representation
        
#lookups, length = create_lookups_from_clicks(data["page"])