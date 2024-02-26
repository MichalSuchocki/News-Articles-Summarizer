from django.shortcuts import render
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
import nltk
from nltk.corpus import stopwords


# Download NLTK stopwords
nltk.download('stopwords')

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_pipeline = pipeline(task="summarization", model=model, tokenizer=tokenizer)
# t5_pipeline = pipeline("text2text-generation", model="flax-community/t5-base-cnn-dm")
# tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-base-cnn-dm")
# model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-base-cnn-dm")

def preprocess_text(text):
    processed_text = re.sub('[,\.!?]', '', text)

    processed_text = processed_text.lower()

    return processed_text


def get_lda_topics(text, num_topics=5):

    words = list(simple_preprocess(text, deacc=True))

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    id2word = gensim.corpora.Dictionary([words])

    corpus = [id2word.doc2bow(words)]

    lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

    topics = lda_model.print_topics(num_topics=num_topics)

    extracted_topics = [re.findall(r'"([^"]*)"', topic[1]) for topic in topics]

    flat_topics = [item for sublist in extracted_topics for item in sublist]

    topics_string = ', '.join(flat_topics)

    return topics_string


def home(request):
    summarizer_ans = None
    header = None
    lda_topics = None

    if request.method == 'POST':
        try:
            input_url = request.POST.get('input_url')

            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--remote-debugging-port=9222')

            driver = webdriver.Chrome(options=chrome_options)

            try:
                driver.get(input_url)
                article_element = driver.find_element(By.CLASS_NAME, 'body')
                header_element = driver.find_element(By.CSS_SELECTOR, "h1")
                article_text = article_element.text
                header = header_element.text

                summarizer_ans = t5_pipeline(article_text, max_length=30000, min_length=100, do_sample=False)[0][
                    'summary_text']
                retrieved_content = article_text

                processed_text = preprocess_text(article_text)

                lda_topics = get_lda_topics(processed_text, num_topics=1)
            except Exception as e:
                error_message = str(e)
                return render(request, 'error.html', {'error_message': error_message})
            finally:
                driver.quit()
        except Exception as e:
            error_message = str(e)
            return render(request, 'error.html', {'error_message': error_message})

    return render(request, 'index.html', {'header': header, 'summarizer_ans': summarizer_ans, 'lda_topics': lda_topics})