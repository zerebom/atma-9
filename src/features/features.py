import pandas as pd
import numpy as np


def annot_category(input_df: pd.DataFrame,
                   master_df: pd.DataFrame):
    """
    カテゴリ ID をひも付けます.

    Args:
        input_df:
            変換するデータ.
            `value_1`  or `JAN` を column として持っている必要があります.
        master_df:
            商品マスタのデータフレーム

    Returns:

    """
    input_df = input_df.rename(columns={'value_1': 'JAN'})
    out_df = pd.merge(input_df['JAN'],
                      master_df[['JAN', 'category_id']], on='JAN', how='left')
    return out_df['category_id']

class AbstractBaseBlock:
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()

class CountEncodingBlock(AbstractBaseBlock):
    """CountEncodingを行なう block"""
    def __init__(self, column: str):
        self.column = column

    def fit(self, input_df, y=None):
        vc = input_df[self.column].value_counts()
        self.count_ = vc
        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.DataFrame()
        out_df[self.column] = input_df[self.column].map(self.count_)
        return out_df.add_prefix('CE_')

class MetaInformationBlock(AbstractBaseBlock):
    def transform(self, input_df):
        use_columns = [
            'hour', 'register_number', 'time_elapsed'
        ]
        return input_df[use_columns].copy()


class DateBlock(AbstractBaseBlock):
    def transform(self, input_df):
        date = pd.to_datetime(input_df['date'])

        out_df = pd.DataFrame({
            'dayofweek': date.dt.dayofweek,
            'day': date.dt.day,
            'year': date.dt.year
        })

        # 金曜日の夜はお祭り騒ぎ
        out_df['hanakin'] = np.where((date.dt.dayofweek == 4) & (input_df['hour'] > 17), 1, 0)

        return out_df.add_prefix('date_')



class HourActionPortfolioBlock(AbstractBaseBlock):
    """時間ごとの `display_action_id` の出現回数を紐付ける block. """
    def __init__(self,dataset):
        self.whole_log = dataset.whole_log
        self.display_action2name = dataset.display_action2name

    def fit(self, input_df, y=None):
        _df = pd.pivot_table(
               data=self.whole_log,
              index='hour',
              columns=self.whole_log['display_action_id'].map(self.display_action2name),
              values='session_id',
              aggfunc='count').fillna(0)

        self.pivot_df_ = _df

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.merge(input_df['hour'],
                          self.pivot_df_,
                          on='hour', how='left').drop(columns=['hour'])

        return out_df.add_prefix('hour_ratio=')

class UserHistoryBlock(AbstractBaseBlock):
    """ユーザーの購買履歴を部門名ごとに集計したベクトルを付与する特徴量 block"""
    def __init__(self,dataset):
        self.whole_log = dataset.whole_log
        self.product_master = dataset.product_master
        self.TARGET_IDS = dataset.target_category_ids
        self.purchase_df = dataset.get_payment_log(self.whole_log)

    def fit(self, input_df, y=None):

        purchase_df = self.purchase_df.rename(columns={ 'value_1': 'JAN' })
        category = annot_category(purchase_df, self.product_master)
        idx_null = category.isnull()  # JAN が紐付かないやつ

        # target の情報はリークになる可能性があるので削除する
        idx_none_target = ~category.isin(self.TARGET_IDS)

        # 商品マスタの部門名を取り出して集計
        bumon_name = pd.merge(purchase_df['JAN'],
                              self.product_master[['JAN', '部門名']], on='JAN', how='left')['部門名']

        _df = pd.pivot_table(data=purchase_df[idx_none_target],
               index='user_id',
               columns=bumon_name[idx_none_target],
              values='n_items',
              aggfunc='sum')\
                .fillna(0)

        # ユーザーごとに平均化.
        _df = _df.div(_df.sum(axis=1), axis=0)

        self.agg_df_ = _df

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.merge(input_df['user_id'], self.agg_df_, on='user_id', how='left').drop(columns=['user_id'])
        out_df = out_df.fillna(0)
        return out_df.add_prefix('ratio_部門名=')


class PublicLogBlock(AbstractBaseBlock):
    """見えているログに関する特徴量"""
    def __init__(self,dataset):
        self.public_log = dataset.public_log

    def fit(self, input_df,y=None):


        self.agg_df_ = pd.concat([
            # 買っている商品の数
            self.public_log.groupby('session_id')['n_items'].sum().rename('total_items'),
            # 買っている商品 (JANレベル) のユニーク数
            self.public_log[self.public_log['kind_1'] == '商品'].groupby('session_id')['value_1'].nunique().rename('JAN_nunique')
        ], axis=1)

        return self.transform(input_df)

    def transform(self, input_df):
        out_df = pd.merge(input_df['session_id'], self.agg_df_, on='session_id', how='left').drop(columns=['session_id'])
        out_df = out_df.fillna(0)
        return out_df.add_prefix('public_log=')