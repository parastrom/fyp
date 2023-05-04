import enum


class QuestionRelations(enum.IntEnum):
    QQ_DIST_2 = 0
    QQ_DIST_1 = 1
    QQ_DIST0 = 2
    QQ_DIST1 = 3
    QQ_DIST2 = 4

class SchemaRelations(enum.IntEnum):
    QC_DEFAULT = 5
    QT_DEFAULT = 6
    CQ_DEFAULT = 7
    CC_DEFAULT = 8
    CT_DEFAULT = 9
    TQ_DEFAULT = 10
    TC_DEFAULT = 11
    TT_DEFAULT = 12


class ColumnRelations(enum.IntEnum):
    CC_FOREIGN_KEY_FORWARD = 13
    CC_FOREIGN_KEY_BACKWARD = 14
    CC_TABLE_MATCH = 15
    CC_DIST_2 = 16
    CC_DIST_1 = 17
    CC_DIST0 = 18
    CC_DIST1 = 19
    CC_DIST2 = 20


class ColumnTableRelations(enum.IntEnum):
    CT_FOREIGN_KEY = 21
    CT_PRIMARY_KEY = 22
    CT_TABLE_MATCH = 23
    CT_ANY_TABLE = 24


class TableRelations(enum.IntEnum):
    TC_TABLE_MATCH = 25
    TC_PRIMARY_KEY = 26
    TC_ANY_TABLE = 27
    TC_FOREIGN_KEY = 28
    TT_FOREIGN_KEY_FORWARD = 29
    TT_FOREIGN_KEY_BACKWARD = 30
    TT_FOREIGN_KEY_BOTH = 31
    TT_DIST_2 = 32
    TT_DIST_1 = 33
    TT_DIST0 = 34
    TT_DIST1 = 35
    TT_DIST2 = 36


class SchemaLinkingRelations(enum.IntEnum):
    QC_CEM = 37
    CQ_CEM = 38
    QT_TEM = 39
    TQ_TEM = 40
    QC_CPM = 41
    CQ_CPM = 42
    QT_TPM = 43
    TQ_TPM = 44


class CellValueRelations(enum.IntEnum):
    QC_NUMBER = 45
    CQ_NUMBER = 46
    QC_TIME = 47
    CQ_TIME = 48
    QC_CELLMATCH = 49
    CQ_CELLMATCH = 50


TOTAL_RELATION_NUM = (
    len(SchemaRelations)
    + len(QuestionRelations)
    + len(ColumnRelations)
    + len(ColumnTableRelations)
    + len(TableRelations)
    + len(SchemaLinkingRelations)
    + len(CellValueRelations)
)

print(TOTAL_RELATION_NUM)