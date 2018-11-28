file_dir = os.path.abspath(os.path.split(__file__)[0])
cache_dir = "cached_data"

def path_for(*p):
	full_path = os.path.join(file_dir, *p)
	full_path_dir = os.path.split(full_path)[0]
	os.makedirs(full_path_dir, exist_ok=True)
	return full_path

class ImageNetDownloader(object):
	api_base = "http://www.image-net.org/api/"

    @staticmethod
	def get_resource(*args,**params):
		args = [ImageNetDownloader.api_base] + list(args)
		return requests.get("/".join(args),params=params).text

    @staticmethod
	def cache_synsets_of_depth(depth, number_per_synset, num_synsets):
		return ImageNetDownloader.cache_images(SynSet.generate_synset_ids_of_depth(depth,num_synsets), number_per_synset)

    @staticmethod
	def cache_images(synset_ids, number_of_images_each):
		return {i: ImageNetDownloader.cache_synset(i, number_of_images_each)[1] for i in synset_ids}

    @staticmethod
	def cache_synset(wnid, limit=100):
		print("Caching %d from synset id %s (%s)" % (limit, wnid, SynSet.get(wnid).word_hierarchy()))
		urls = ImageNetDownloader.get_resource('text','imagenet.synset.geturls', wnid=[wnid])
		all_urls = urls.splitlines()
		shuffle(all_urls)
		n = min(len(all_urls), limit)
		image_urls = all_urls
		url_lookup = {}
		num_saved = 0
		files_written = []
		for i in range(len(image_urls)):
			if num_saved >= limit:
				break
			u = image_urls[i].strip()
			file_ending = urllib.parse.urlparse(u).path.split('.')[-1].strip()
			fp = path_for(cache_dir, wnid, "%d.%s" % (num_saved,file_ending))
			fp = fp.strip()
			valid_image = False
			try:
				resp = requests.get(u)
				if resp.url.lower() == u.lower():
					downloaded_im = resp.content
					if imghdr.what('',h=downloaded_im) is not None:
						# valid image
						valid_image = True
						with open(fp,'wb') as output_file:
							output_file.write(downloaded_im)
						num_saved += 1
				else:
					print("Expected",u,"Got",resp.url)
			except Exception as err:
				#print(err)
				pass
			if valid_image:
				files_written.append(fp)
				url_lookup[fp] = u
		with open(path_for(cache_dir,wnid,'credit.txt'),'w') as credit_file:
			json.dump({"wnid":wnid,"words":SynSet.get(wnid).word_hierarchy(),"urls":url_lookup},credit_file,indent=2)
		return wnid, files_written

	@staticmethod
	def clear_cache():
		p = path_for(cache_dir)
		if input("Deleting %s (y/n) - " % p).lower().startswith('y'):
			shutil.rmtree(path_for(cache_dir))
		else:
			print("Not deleting")

class SynSet(object):
	lookup = {}
	child_lookup = {}
	def __init__(self, wnid, words, has_images, parent_wnid = None):
		self.wnid = wnid
		self.words = words
		self.has_images = has_images
		self.parent_wnid = parent_wnid
		self.depth = 0
		if self.wnid is not None:
			SynSet.lookup[wnid] = self
			if self.parent_wnid not in SynSet.child_lookup:
				SynSet.child_lookup[self.parent_wnid] = []
			SynSet.child_lookup[self.parent_wnid].append(self.wnid)
		if self.parent_wnid in SynSet.lookup:
			self.depth = SynSet.lookup[self.parent_wnid].depth + 1

	def root():
		return SynSet.child_lookup[None]

	def all_sets():
		return list(SynSet.lookup.values())

	def all_sets_with_images():
		return filter(lambda s: s.has_images, SynSet.all_sets())

	def clear_lookup():
		SynSet.lookup.clear()
	def get(k):
		return SynSet.lookup[k]
	def __str__(self):
		return "Synset %s"% (str(self.wnid))
	def __repr__(self):
		return str(self)
	def hierarchy(self):
		if self.parent_wnid is None:
			return str(self.wnid)
		else:
			return "%s > %s" % (SynSet.lookup[self.parent_wnid].hierarchy(), str(self.wnid))
	def word_hierarchy(self):
		if self.parent_wnid is None:
			return str(self.words)
		else:
			return "%s > %s" % (SynSet.lookup[self.parent_wnid].word_hierarchy(), str(self.words))
	def generate_synset_ids_of_depth(depth, limit = None):
		all_syn = list(map(lambda s: s.wnid, filter(lambda s: s.depth == depth, SynSet.all_sets_with_images())))
		if limit is not None:
			shuffle(all_syn)
			all_syn = all_syn[:limit]
		return all_syn

	def populate_structure():
		SynSet.clear_lookup()
		valid_synsets = set(requests.get('http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list').text.splitlines())
		structure_file_contents = None
		with open(path_for("structure.xml"), "r") as f:
			structure_file_contents = f.read()
		soup = Soup(structure_file_contents,'lxml').imagenetstructure
		q = collections.deque()
		q.append((soup, None))
		while len(q) > 0:
			parent_soup, parent_wnid = q.popleft()
			try:
				my_wnid = parent_soup['wnid']
			except:
				my_wnid = None
			try:
				my_words=parent_soup['words']
			except:
				my_words=""
			if my_wnid is not None:
				n = SynSet(my_wnid, my_words, my_wnid in valid_synsets, parent_wnid)
			child_synsets = [(i,my_wnid) for i in parent_soup.find_all('synset',recursive=False)]
			q.extend(child_synsets)
		return SynSet.lookup
