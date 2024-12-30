import streamlit as st
import requests
import re
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import seaborn as sns
import validators

# Set the font path, ensure the path points to a valid Chinese font
font_path = 'Font/SimHei.ttf'  # Please modify according to the actual font path
font_prop = fm.FontProperties(fname=font_path)

# URL fetch function
def fetch_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check if the request was successful
    response.encoding = 'utf-8'
    return response.text

def remove_html_tags(html):
    """Remove HTML tags"""
    return re.sub(r'<[^>]+>', '', html)

def remove_punctuation_and_english(text):
    """Remove punctuation and English characters, keep Chinese characters"""
    return re.sub(r'[A-Za-z0-9\s+]|[^\u4e00-\u9fa5]+', '', text)

# Generate word cloud
def generate_wordcloud(word_counts):
    wordcloud = WordCloud(font_path=font_path, width=400, height=200,
                          background_color='white').generate_from_frequencies(word_counts)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Do not show the axes
    fig = plt.gcf()
    return fig

def main():
    st.title("文本分析与词频可视化")

    # Side bar for stop words file upload
    stopwords_file = st.sidebar.file_uploader("上传停用词文件 (stopwords.txt):", type=['txt'])

    # Side bar for URL input
    url = st.sidebar.text_input("输入文章的 URL:", placeholder="https://example.com")

    # Select chart type
    chart_type = st.sidebar.selectbox("选择可视化图型:",
                                       ["柱状图", "饼图", "折线图", "面积图", "散点图", "箱线图", "词云图"])

    if st.sidebar.button("抓取并分析"):
        # Validate the input URL
        if not validators.url(url):
            st.error("请输入有效的 URL!")
            return

        # Read stop words
        if stopwords_file is not None:
            stopwords = set(stopwords_file.read().decode('utf-8').splitlines())
        else:
            st.error("请上传停用词文件!")
            return

        try:
            # Fetch HTML content
            html = fetch_content(url)
            clean_text = remove_html_tags(html)  # Remove HTML tags
            clean_text = remove_punctuation_and_english(clean_text)  # Remove punctuation and English
            words = jieba.lcut(clean_text)  # Perform word segmentation

            # Filter meaningful words
            meaningful_words = [word for word in words if word not in stopwords and len(word) > 1]
            word_counts = Counter(meaningful_words)  # Count word frequencies
            most_common_words = word_counts.most_common(20)  # Get the top 20 frequent words
            freq_df = pd.DataFrame(most_common_words, columns=['词语', '频率'])

            # Expandable section for the extracted article
            with st.expander("点击查看提取的文章"):
                st.write(clean_text)

            # Word frequency display
            st.write("词频排名前 20 的词汇：")
            st.dataframe(freq_df)

            # Generate corresponding graphs based on selected type
            if chart_type == "柱状图":
                fig, ax = plt.subplots()
                bars = ax.bar(freq_df['词语'], freq_df['频率'], color='orange')
                plt.title('词频最高的 20 个词', fontproperties=font_prop)
                plt.xlabel('词语', fontproperties=font_prop)
                plt.ylabel('频率', fontproperties=font_prop)
                plt.xticks(rotation=45, fontproperties=font_prop)
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 1), ha='center', va='bottom')
                st.pyplot(fig)

            elif chart_type == "饼图":
                fig1, ax1 = plt.subplots()
                wedges, texts, autotexts = ax1.pie(
                    freq_df['频率'], labels=freq_df['词语'], autopct='%1.1f%%', startangle=90)
                plt.axis('equal')  # Make pie chart a circle
                plt.title('词频饼图', fontproperties=font_prop)
                # Set the fonts in pie chart
                for text in texts + autotexts:
                    text.set_fontproperties(font_prop)
                st.pyplot(fig1)

            elif chart_type == "折线图":
                fig2, ax2 = plt.subplots()
                ax2.plot(freq_df['词语'], freq_df['频率'], marker='o', color='b')
                plt.title('词频折线图', fontproperties=font_prop)
                plt.xlabel('词语', fontproperties=font_prop)
                plt.ylabel('频率', fontproperties=font_prop)
                plt.xticks(rotation=45, fontproperties=font_prop)
                for i, txt in enumerate(freq_df['频率']):
                    ax2.annotate(txt, (freq_df['词语'][i], freq_df['频率'][i]), textcoords="offset points",
                                 xytext=(0, 10), ha='center')
                st.pyplot(fig2)

            elif chart_type == "面积图":
                fig3, ax3 = plt.subplots()
                ax3.fill_between(freq_df['词语'], freq_df['频率'], color='skyblue', alpha=0.5)
                plt.title('词频面积图', fontproperties=font_prop)
                plt.xlabel('词语', fontproperties=font_prop)
                plt.ylabel('频率', fontproperties=font_prop)
                plt.xticks(rotation=45, fontproperties=font_prop)
                for x, y in zip(freq_df['词语'], freq_df['频率']):
                    ax3.text(x, y, str(y), ha='center', va='bottom')
                st.pyplot(fig3)

            elif chart_type == "散点图":
                fig4, ax4 = plt.subplots()
                ax4.scatter(freq_df['词语'], freq_df['频率'], color='red')
                plt.title('词频散点图', fontproperties=font_prop)
                plt.xlabel('词语', fontproperties=font_prop)
                plt.ylabel('频率', fontproperties=font_prop)
                plt.xticks(rotation=45, fontproperties=font_prop)
                for i, txt in enumerate(freq_df['频率']):
                    ax4.annotate(txt, (freq_df['词语'][i], freq_df['频率'][i]), textcoords="offset points",
                                  xytext=(0, 10), ha='center')
                st.pyplot(fig4)

            elif chart_type == "箱线图":
                fig5, ax5 = plt.subplots()
                sns.boxplot(x=freq_df['频率'], ax=ax5)
                plt.title('词频箱线图', fontproperties=font_prop)
                plt.xlabel('频率', fontproperties=font_prop)
                st.pyplot(fig5)

            elif chart_type == "词云图":
                st.write("生成词云图：")
                wordcloud_fig = generate_wordcloud(word_counts)
                st.pyplot(wordcloud_fig)  # Display word cloud

        except Exception as e:
            st.error(f"发生错误：{e}")

if __name__ == "__main__":
    main()