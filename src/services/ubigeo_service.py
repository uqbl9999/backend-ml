"""
Ubigeo Service
Handles mapping between Department+Province and Ubigeo codes
"""

import pandas as pd
import os
from typing import Dict, List, Optional


class UbigeoService:
    """Service to handle ubigeo mappings"""

    def __init__(self, ubigeo_file_path: str):
        """
        Initialize the ubigeo service

        Parameters:
        -----------
        ubigeo_file_path : str
            Path to the TB_UBIGEOS.csv file
        """
        self.ubigeo_file_path = ubigeo_file_path
        self.df_ubigeos = None
        self.load_ubigeos()

    def load_ubigeos(self):
        """Load the ubigeos CSV file"""
        try:
            # Try different encodings
            encodings = ['iso-8859-1', 'latin-1', 'cp1252', 'utf-8']

            for encoding in encodings:
                try:
                    # Read the file line by line to handle the special format
                    # Each line is wrapped in quotes: "field1\tfield2\tfield3"
                    lines = []
                    with open(self.ubigeo_file_path, 'r', encoding=encoding) as f:
                        for line in f:
                            # Remove quotes and trailing newline
                            line = line.strip().strip('"')
                            # Split by tab
                            fields = line.split('\t')
                            # Take only first 4 fields (ubigeo, departamento, provincia, distrito)
                            if len(fields) >= 4:
                                lines.append(fields[:4])

                    # Create DataFrame
                    if len(lines) > 1:
                        self.df_ubigeos = pd.DataFrame(
                            lines[1:],  # Skip header
                            columns=['ubigeo_reniec', 'departamento', 'provincia', 'distrito']
                        )
                        print(f"✅ Loaded ubigeos with encoding: {encoding}")
                        break
                except (UnicodeDecodeError, Exception) as e:
                    continue

            if self.df_ubigeos is None:
                raise Exception("Could not read file with any encoding")

            # Clean data (remove extra spaces and convert to uppercase)
            self.df_ubigeos['ubigeo_reniec'] = self.df_ubigeos['ubigeo_reniec'].str.strip()
            self.df_ubigeos['departamento'] = self.df_ubigeos['departamento'].str.strip().str.upper()
            self.df_ubigeos['provincia'] = self.df_ubigeos['provincia'].str.strip().str.upper()
            self.df_ubigeos['distrito'] = self.df_ubigeos['distrito'].str.strip().str.upper()

            # Filter out rows with empty ubigeo
            self.df_ubigeos = self.df_ubigeos[self.df_ubigeos['ubigeo_reniec'] != '']

            # Convert ubigeo to integer
            self.df_ubigeos['ubigeo_reniec'] = self.df_ubigeos['ubigeo_reniec'].astype(int)

            print(f"✅ Ubigeos loaded: {len(self.df_ubigeos)} records")

        except Exception as e:
            print(f"❌ Error loading ubigeos: {e}")
            raise

    def get_provincias_by_departamento(self, departamento: str) -> List[str]:
        """
        Get list of provinces for a given department

        Parameters:
        -----------
        departamento : str
            Department name

        Returns:
        --------
        List[str] : List of province names
        """
        departamento = departamento.strip().upper()

        provincias = self.df_ubigeos[
            self.df_ubigeos['departamento'] == departamento
        ]['provincia'].unique().tolist()

        return sorted(provincias)

    def get_ubigeo_by_dept_prov(self, departamento: str, provincia: str) -> Optional[int]:
        """
        Get ubigeo code for a given department and province
        Returns the first district's ubigeo for that province (usually the capital)

        Parameters:
        -----------
        departamento : str
            Department name
        provincia : str
            Province name

        Returns:
        --------
        Optional[int] : Ubigeo code or None if not found
        """
        departamento = departamento.strip().upper()
        provincia = provincia.strip().upper()

        # Filter by department and province
        matches = self.df_ubigeos[
            (self.df_ubigeos['departamento'] == departamento) &
            (self.df_ubigeos['provincia'] == provincia)
        ]

        if len(matches) == 0:
            return None

        # Return the first ubigeo (usually the capital district)
        # The ubigeo format is DDPPDD where:
        # - DD = department code
        # - PP = province code
        # - DD = district code
        # We want the one ending in '01' (capital district) if available

        # Try to find the capital district (ends with 01)
        capital = matches[matches['ubigeo_reniec'] % 100 == 1]
        if len(capital) > 0:
            return int(capital.iloc[0]['ubigeo_reniec'])

        # Otherwise return the first one
        return int(matches.iloc[0]['ubigeo_reniec'])

    def get_all_departamentos(self) -> List[str]:
        """
        Get list of all departments

        Returns:
        --------
        List[str] : List of department names
        """
        return sorted(self.df_ubigeos['departamento'].unique().tolist())

    def validate_departamento_provincia(self, departamento: str, provincia: str) -> bool:
        """
        Validate if a department-province combination exists

        Parameters:
        -----------
        departamento : str
            Department name
        provincia : str
            Province name

        Returns:
        --------
        bool : True if valid combination, False otherwise
        """
        ubigeo = self.get_ubigeo_by_dept_prov(departamento, provincia)
        return ubigeo is not None

    def get_location_info(self, ubigeo: int) -> Optional[Dict]:
        """
        Get location information for a given ubigeo

        Parameters:
        -----------
        ubigeo : int
            Ubigeo code

        Returns:
        --------
        Optional[Dict] : Location information or None if not found
        """
        match = self.df_ubigeos[self.df_ubigeos['ubigeo_reniec'] == ubigeo]

        if len(match) == 0:
            return None

        row = match.iloc[0]
        return {
            'ubigeo': int(row['ubigeo_reniec']),
            'departamento': row['departamento'],
            'provincia': row['provincia'],
            'distrito': row['distrito']
        }


# Singleton instance
_ubigeo_service = None


def get_ubigeo_service(ubigeo_file_path: str = None) -> UbigeoService:
    """
    Get or create the ubigeo service singleton

    Parameters:
    -----------
    ubigeo_file_path : str, optional
        Path to the ubigeos file (only needed on first call)

    Returns:
    --------
    UbigeoService : The ubigeo service instance
    """
    global _ubigeo_service

    if _ubigeo_service is None:
        if ubigeo_file_path is None:
            # Default path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ubigeo_file_path = os.path.join(base_dir, 'data', 'TB_UBIGEOS.csv')

        _ubigeo_service = UbigeoService(ubigeo_file_path)

    return _ubigeo_service
