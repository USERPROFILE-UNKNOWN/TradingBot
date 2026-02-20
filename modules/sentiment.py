import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

class NewsSentinel:
    def __init__(self, api_interface):
        self.api = api_interface
        self.analyzer = SentimentIntensityAnalyzer()
        
        # v4.1: Inject Financial Lexicon
        # VADER defaults: -4.0 (Most Negative) to +4.0 (Most Positive)
        financial_lexicon = {
            'beat': 3.0, 'missed': -3.0, 'surpassed': 2.5, 'fell short': -2.5,
            'upgrade': 3.5, 'downgrade': -3.5, 'buy rating': 3.0, 'sell rating': -3.0,
            'merger': 2.0, 'acquisition': 2.0, 'bankruptcy': -4.0, 'insolvency': -4.0,
            'fda approval': 4.0, 'fda rejection': -4.0, 'lawsuit': -3.0, 'settlement': 1.0,
            'guidance raised': 3.0, 'guidance lowered': -3.0, 'dividend hike': 2.5, 'dividend cut': -3.5,
            'breakout': 2.0, 'plummet': -3.0, 'surge': 2.5, 'plunge': -3.0,
            'all-time high': 2.0, 'all-time low': -2.0, 'record revenue': 3.0
        }
        self.analyzer.lexicon.update(financial_lexicon)
        
        # Cache: {symbol: {'score': float, 'time': datetime}}
        self.cache = {} 
        self.cache_duration = timedelta(minutes=15) # Re-scan news every 15 mins

    def get_sentiment(self, symbol, lookback_hours=24):
        """
        Fetches news and returns a score (-1.0 to 1.0) and label.
        Uses caching to prevent API throttling.
        """
        try:
            # Clean Symbol
            search_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Check Cache
            if search_symbol in self.cache:
                last_check = self.cache[search_symbol]['time']
                if datetime.now() - last_check < self.cache_duration:
                    return self.cache[search_symbol]['score'], self.cache[search_symbol]['label']

            end_dt = datetime.now()
            start_dt = end_dt - timedelta(hours=lookback_hours)
            
            # Fetch News
            news = self.api.get_news(symbol=search_symbol, 
                                     start=start_dt.strftime('%Y-%m-%d'), 
                                     limit=15) # Increased limit
            
            if not news:
                self._update_cache(search_symbol, 0.0, "No News")
                return 0.0, "No News"
            
            total_score = 0
            count = 0
            
            for article in news:
                # Weight Headlines higher than Summary
                headline_score = self.analyzer.polarity_scores(article.headline)['compound']
                summary_score = self.analyzer.polarity_scores(article.summary)['compound']
                
                # Weighted Average (Headline 60%, Summary 40%)
                composite = (headline_score * 0.6) + (summary_score * 0.4)
                
                # Recency Weighting (Newer news matters more)
                # Note: Alpaca news objects technically have timestamps, simplified here
                total_score += composite
                count += 1
                
            if count == 0: 
                self._update_cache(search_symbol, 0.0, "Neutral")
                return 0.0, "Neutral"
            
            avg_score = total_score / count
            label = self.interpret_score(avg_score)
            
            self._update_cache(search_symbol, avg_score, label)
            return avg_score, label
            
        except Exception as e:
            # print(f"Sentiment Error: {e}") # Silent error
            return 0.0, "Error"

    def _update_cache(self, symbol, score, label):
        self.cache[symbol] = {
            'score': score,
            'label': label,
            'time': datetime.now()
        }

    def interpret_score(self, score):
        if score > 0.5: return "Very Bullish"
        if score > 0.1: return "Bullish"
        if score < -0.5: return "Very Bearish"
        if score < -0.1: return "Bearish"
        return "Neutral"