
import pathlib
import numpy as np
import pandas as pd

class RetailDataset:
    def __init__(self, file_path: pathlib.Path, thres_sec: int) -> None:
        self.file_path = file_path
        self.thres_sec = thres_sec
        self.cartlog: pd.DataFrame = pd.read_pickle(file_path / "cartlog.pkl")
        self.product_master: pd.DataFrame = pd.read_pickle(
            file_path / "product_master.pkl"
        )

        self.product_master = self.product_master.rename(
            columns={'jan_code': "JAN"})
        self.product_master['JAN'] = self.product_master['JAN'].astype(str)

        self.user_master: pd.DataFrame = pd.read_pickle(
            file_path / "user_master.pkl")
        self.meta: pd.DataFrame = pd.read_pickle(file_path / "meta.pkl")
        self.test: pd.DataFrame = pd.read_pickle(file_path / "test.pkl")
        self.display_action: pd.DataFrame = pd.read_pickle(
            file_path / "display_action_id.pkl")

        self.meta.loc[
            self.meta["time_elapsed_sec"].isnull(), "time_elapsed_sec"
        ] = thres_sec
        self.target_category_ids = [
            38,  # アイスクリーム__ノベルティー
            110,  # スナック・キャンディー__ガム
            113,  # スナック・キャンディー__シリアル
            114,  # スナック・キャンディー__スナック
            134,  # チョコ・ビスクラ__チョコレート
            171,  # ビール系__RTD
            172,  # ビール系__ノンアルコール
            173,  # ビール系__ビール系
            376,  # 和菓子__米菓
            435,  # 大型PET__無糖茶（大型PET）
            467,  # 小型PET__コーヒー（小型PET）
            537,  # 水・炭酸水__大型PET（炭酸水）
            539,  # 水・炭酸水__小型PET（炭酸水）
            629,  # 缶飲料__コーヒー（缶）
            768,  # 麺類__カップ麺
        ]

        self.target_category = [
            # お酒に関するもの
            'ビール系__RTD', 'ビール系__ビール系', 'ビール系__ノンアルコール',

            # お菓子に関するもの
            'スナック・キャンディー__スナック',
            'チョコ・ビスクラ__チョコレート',
            'スナック・キャンディー__ガム',
            'スナック・キャンディー__シリアル',
            'アイスクリーム__ノベルティー',
            '和菓子__米菓',

            # 飲料に関するもの
            '水・炭酸水__大型PET（炭酸水）',
            '水・炭酸水__小型PET（炭酸水）',
            '缶飲料__コーヒー（缶）',
            '小型PET__コーヒー（小型PET）',
            '大型PET__無糖茶（大型PET）',

            # 麺類
            '麺類__カップ麺',
        ]

    def prepare_data(self):
        idx_test = self.cartlog['session_id'].isin(self.get_test_sessions())
        self.whole_log = self.cartlog[~idx_test].reset_index(drop=True)
        self.display_action2name = self.get_display_action2name()

        # train_whole_log_df
        # self.whole_train = self.get_train_output_log()
        # self.train_public, self.train_private = self.get_train_input_log()
        # self.public_log = pd.concat([self.train_public,self.get_test_input_log()],axis=0,ignore_index=True)

    def get_display_action2name(self):
        display_action_names = self.display_action['display_name'] + \
            '_' + self.display_action['action_name']
        return dict(zip(self.display_action['display_action_id'], display_action_names))

    def get_category_id2code(self):
        cat2id = dict(
            zip(self.product_master['category_name'], self.product_master['category_id']))
        TARGET_IDS = pd.Series(self.target_category).map(
            cat2id).values.tolist()
        category_id2code = dict(zip(TARGET_IDS, self.target_category))
        return category_id2code

    def get_test_sessions(self) -> set:
        """以下の条件を満たすセッションを取得する
        - 予測対象である
        """
        return set(self.test["session_id"].unique())

    def get_test_input_log(self) -> pd.DataFrame:
        """以下の条件を満たすログを取得する
        - 予測対象である

        ログが存在しないセッションもあるので注意.
        """
        test_sessions = self.get_test_sessions()
        return self.cartlog[self.cartlog["session_id"].isin(test_sessions)]

    def get_log_first_half(self) -> pd.DataFrame:
        """以下の条件を満たすログを取得する
        - 学習期間(2020-08-01の前日まで)のセッションである
        """
        first_half_sessions = set(
            self.meta.query("date < '2020-08-01'")["session_id"].unique()
        )
        return self.cartlog[self.cartlog["session_id"].isin(first_half_sessions)]

    def get_train_output_log(self) -> pd.DataFrame:
        """以下の条件を満たすログを取得する
        - 学習期間(2020-08-01の前日まで)のセッションである
        - 指定した時間(thres_sec)以降にログが存在している
        推論したいデータ
        """
        return pd.merge(
            self.get_log_first_half(),
            self.meta[["session_id", "time_elapsed_sec"]],
            on=["session_id"],
            how="inner",
        ).query("spend_time > time_elapsed_sec")

    def get_train_sessions(self) -> set:
        """以下の条件を満たすセッションを取得する
        - 学習期間(2020-08-01の前日まで)のセッションである
        - 指定した時間(thres_sec)以降にログが存在している
        """
        return set(self.get_train_output_log()["session_id"].unique())

    def get_train_input_log(self) -> pd.DataFrame:
        """以下の条件を満たすログを取得する
        - 学習期間(2020-08-01の前日まで)のセッションである
        - 指定した時間(thres_sec)以降にログが存在している
        - 指定した時間(thres_sec)より前のログである
        """
        train_sessions = self.get_train_sessions()

        train_whole_log = pd.merge(
            self.get_log_first_half()[
                self.get_log_first_half()["session_id"].isin(train_sessions)
            ],
            self.meta[["session_id", "time_elapsed_sec"]],
            on=["session_id"],
            how="inner",
        )
        train_public = train_whole_log.query("spend_time <= time_elapsed_sec")
        train_private = train_whole_log.query("spend_time > time_elapsed_sec")
        return train_public, train_private

    def get_payment_sessions(self) -> set:
        """以下の条件を満たすセッションを取得する
        - 決済を行った
        """
        return set(self.cartlog.query("is_payment == 1")["session_id"].unique())

    def get_payment_log(self, input_df) -> pd.DataFrame:
        idx = input_df['kind_1'] == '商品'
        out_df = input_df[idx].reset_index(drop=True)
        return out_df

    def agg_payment(self, cartlog) -> pd.DataFrame:
        """セッションごと・商品ごとの購買個数を集計する"""
        # 購買情報は商品のものだけ.
        target_index = cartlog["kind_1"] == "商品"

        # JANコード (vale_1)ごとに商品の購入個数(n_items)を足し算
        agg = (
            cartlog.loc[target_index]
            .groupby(["session_id", "value_1"])["n_items"]
            .sum()
            .reset_index()
        )
        agg = agg.rename(columns={"value_1": "jan_code"})
        agg = agg.astype({"jan_code": int})
        return agg

    def get_train_target(self) -> pd.DataFrame:
        """学習で使用するセッションの目的変数を取得する"""
        # 空のターゲット用データフレームを用意する
        train_sessions = self.get_train_sessions()
        train_target = pd.DataFrame(
            np.zeros((len(train_sessions), len(self.target_category_ids))),
            index=train_sessions,
            columns=self.target_category_ids,
        ).astype(int)
        train_target.index.name = "session_id"

        # 集計する
        train_output_log = self.get_train_output_log()
        train_items_per_session_jan = self.agg_payment(train_output_log)
        train_items_per_session_target_jan = pd.merge(
            train_items_per_session_jan,
            self.product_master[["jan_code", "category_id"]],
            on="jan_code",
            how="inner",
        ).query("category_id in @self.target_category_ids")
        train_target_pos = (
            train_items_per_session_target_jan.groupby(["session_id", "category_id"])[
                "n_items"
            ]
            .sum()
            .unstack()
            .fillna(0)
            .astype(int)
        )
        train_target_pos[train_target_pos > 0] = 1
        train_target_pos[train_target_pos <= 0] = 0

        train_target.loc[train_target_pos.index] = train_target_pos.values
        return train_target[self.target_category_ids]


def only_purchase_records(input_df: pd.DataFrame) -> pd.DataFrame:
    idx = input_df['kind_1'] == '商品'
    out_df = input_df[idx].reset_index(drop=True)
    return out_df


def create_payment(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    ログデータから session_id / JAN ごとの購買情報に変換します.

    Args:
        input_df:
            レジカートログデータ

    Returns:
        session_id, JAN, n_items (合計購買数) の DataFrame
    """

    # 購買情報は商品のものだけ.
    out_df = only_purchase_records(input_df)
    out_df = out_df.groupby(['session_id', 'value_1'])[
        'n_items'].sum().reset_index()
    out_df = out_df.rename(columns={
        'value_1': 'JAN'
    })
    return out_df


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


def only_payment_session_record(input_log_df):
    """支払いが紐づくセッションへ絞り込みを行なう"""
    payed_sessions = input_log_df[input_log_df['is_payment']
                                  == 1]['session_id'].unique()
    idx = input_log_df['session_id'].isin(payed_sessions)
    out_df = input_log_df[idx].reset_index(drop=True)
    return out_df


def create_target_from_log(log_df: pd.DataFrame,
                           product_master_df: pd.DataFrame,
                           TARGET_IDS,
                           only_payment=True):

    if only_payment:
        log_df = only_payment_session_record(log_df)
    pay_df = create_payment(log_df)
    pay_df['category_id'] = annot_category(pay_df, master_df=product_master_df)

    # null の category を削除. JAN が紐付かない時に発生する.
    idx_null = pay_df['category_id'].isnull()
    pay_df = pay_df[~idx_null].reset_index(drop=True)
    # Nullが混じっている時 float になるため int へ明示的に戻す
    pay_df['category_id'] = pay_df['category_id'].astype(int)

    idx = pay_df['category_id'].isin(TARGET_IDS)
    target_df = pd.pivot_table(data=pay_df[idx],
                               index='session_id',
                               columns='category_id',
                               values='n_items',
                               aggfunc='sum')

    sessions = sorted(log_df['session_id'].unique())
    print(len(sessions))
    target_df = target_df.reindex(sessions)
    target_df = target_df.fillna(0).astype(int)
    return target_df, pay_df
