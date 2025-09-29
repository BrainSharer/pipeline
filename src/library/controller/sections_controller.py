import re
from library.database_model.slide import Section

class SectionsController():
    """Class for controlling sections
    """

    def get_sections(self, animal: str, channel: int, debug: bool = False) -> list:
        """The sections table is a view and it is already filtered by active and file_status = 'good'
        The ordering is important. This needs to come from the scan run table.
        It used to be the histology table, but Song-mao needed a way to reoder them manually.

        :param animal: the animal to query
        :param channel: 1 or 2 or 3.
        :param debug: whether to print the raw SQL query

        :returns: list of sections in order
        """
        slide_orderby = self.scanrun.slide_order
        slide_orderby = slide_orderby.lower().strip()

        if slide_orderby == 'desc':
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.desc())\
                .order_by(Section.scene_number.desc())
        elif slide_orderby == 'manual':
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.scene_order.asc())
        else:
            print('Using default order by')
            query = self.session.query(Section).filter(Section.prep_id == animal)\
                .filter(Section.channel == channel)\
                .order_by(Section.slide_physical_id.asc())\
                .order_by(Section.scene_number.asc())

        if debug: # Print the raw SQL query
            # raw_sql = str(query.statement.compile(compile_kwargs={"literal_binds": True}))
            # cleaned_sql = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', r'\1', raw_sql)
            raw_sql = str(query.statement.compile(compile_kwargs={"literal_binds": True}))
            cleaned_sql = self.ensure_backticks(raw_sql)
            print(f'RAW SQL: {cleaned_sql}')
            print(f'RAW SQL: \n{cleaned_sql}\n')

        sections = query.all()
        return sections


    def ensure_backticks(self, sql_string):
        """
        Ensure all identifiers have backticks, but exclude SQL keywords
        FOR DEBUG OF RAW SQL
        """
        # Remove existing quotes and backticks
        sql_string = sql_string.replace('`', '').replace('"', '')
        
        # Add backticks to ALL table.column patterns
        sql_string = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)', r'`\1`.`\2`', sql_string)
        
        # Add backticks to standalone table names in FROM clause
        sql_string = re.sub(r'\b(FROM|JOIN|INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'\1 `\2`', sql_string)
        
        return sql_string


    def get_section_count(self, animal):
        count = self.session.query(Section)\
            .filter(Section.prep_id == animal)\
            .filter(Section.channel == 1)\
            .count() 
        return count