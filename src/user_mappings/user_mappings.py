"""
Author: Jan Koci

Implementation of class UserMappings that maps users to unique identifiers.

Each user in the dataset is represented as a tuple (visitor_id, [user_id])
where the user_id is optional. The visitor_id represents user's cookies
and user_id is an identifier of logged users.
The class implements these rules:

    1. It is able to connect interactions where only visitor_id is known (visitor_id,)
       to the actual user, once he logges in and creates interactions (visitor_id, user_id)

    2. It can represent users both by their visitor_id and their user_id
       however user_id has bigger priority

    3. It handles cases where one user_id is used with more visitor_ids

    4. It handles cases where one visitor_id is used by more than one user_id

"""
from progress.bar import Bar


def add_uid_to_df(df, user_mappings):
    ids = list()
    for _, row in df.iterrows():
        user = user_mappings.get_user(row.visitor_id, row.user_id)
        if (type(user) == list or not user):
            ids.append(-1)
        else:
            ids.append(user)
    df = df.assign(uid=ids)
    return df


def create_user_mappings(df):
    mappings = UserMappings()
    bar = Bar("Processing", max=df.shape[0])

    for _, row in df.iterrows():
        visitor_id = row.visitor_id
        user_id = row.user_id

        user = mappings.get_user(visitor_id, user_id)

        if (user == None):
            mappings.add_user(visitor_id, user_id)

        else:
            mappings.actualize_user(visitor_id, user_id)
        bar.next()

    bar.finish()
    return mappings


class UserMappings():
    """Class that maps users represented as tuples (visitor_id, [user_id])
    to unique identifiers

    Attributes:
        mappings    : dictionary where keys are either users' user_ids or visitor_ids
                      and values are their unique identifiers
        pointers    : dictionary where keys are visitor_ids and values are lists
                      with user_ids that were used together with them

    Interactions where only visitor_id is known are stored in the mappings dictionary.
    With interactions where both visitor_id and user_id are known, the user_id
    is stored in the mappings dictionary and the visitor_id in the pointers dictionary
    as a key, its value is a list containing the user_id.
    Here we visualize the above example:

        First interaction, only visitor_id is known => (vid_1,)
        Second interaction, both visitor_id and user_id are known => (vid_2, uid_2)

        mappings = {vid_1 : 0, uid_2 : 1}
        pointers = {vid_2 : [uid_2]}

    """

    def __init__(self):
        self.__mappings = {}
        self.__pointers = {}

    @property
    def mappings(self):
        return self.__mappings

    @property
    def pointers(self):
        return self.__pointers


    def add_user(self, visitor_id, user_id=''):
        """Adds new user to mappings
        If only the visitor_id is known it will be added to self.__mappings
        if both visitor_id and user_id are known the user_id will be added
        to self.__mappings and visitor_id to self.__pointers pointing
        to the user_id

        """
        if self.get_user(visitor_id, user_id) != None:
            print('[Warning] User already in mappings')
            return # THROW !!!

        if user_id:
            self.__mappings[user_id] = len(self.__mappings)
            if visitor_id in self.__pointers:
                # print('[BLACK MAGIC] - interesting')
                self.__pointers[visitor_id].append(user_id)
                return
            else:
                self.__pointers[visitor_id] = list()
                self.__pointers[visitor_id].append(user_id)
        else:
            if visitor_id in self.__pointers:
                return
            self.__mappings[visitor_id] = len(self.__mappings)


    def get_user(self, visitor_id, user_id=''):
        """Returns ID of the user stored in self.__mappings
        If user is not in the mappings it returns None
        If no user_id is passed and the visitor_id matches more users
        it returns list of their user_ids stored in self.__pointers

        """
        if user_id:
            if user_id in self.__mappings:
                return self.__mappings[user_id]

            if visitor_id in self.__mappings:
                return self.__mappings[visitor_id]
        else:
            if visitor_id in self.__pointers:
                pointers = self.__pointers[visitor_id]

                if len(pointers) == 1:
                    return self.__mappings[pointers[0]]
                else:
                    return pointers

            elif visitor_id in self.__mappings:
                return self.__mappings[visitor_id]


    def actualize_user(self, visitor_id, user_id=''):
        """Actualize data stored for the user
        This method can:
            1. replace visitor_id in mappings with user_id and move it to pointers
            2. append user_id to visitor_id's list of known users in pointers
            3. add new visitor_id to pointers
            4. add new user_id to mappings

        """
        if user_id:
            if user_id in self.__mappings:
                if visitor_id in self.__pointers:
                    if not (user_id in self.__pointers[visitor_id]):
                        self.__pointers[visitor_id].append(user_id)
                else:
                    if visitor_id in self.__mappings:
                        del self.__mappings[visitor_id]

                    self.__pointers[visitor_id] = list()
                    self.__pointers[visitor_id].append(user_id)
            else:
                if visitor_id in self.__mappings:
                    self.__mappings[user_id] = self.__mappings.pop(visitor_id)
                    self.__pointers[visitor_id] = list()
                    self.__pointers[visitor_id].append(user_id)
