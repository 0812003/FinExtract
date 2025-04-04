
import React, { useState } from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
    const [menuOpen, setMenuOpen] = useState(false);

    const toggleMenu = () => {
        setMenuOpen(!menuOpen);
    };

    return (
        <div className="py-6 px-10 ">
            <header className="bg-teal-50 border-2 text-black shadow-xl rounded-xl">
                <div className="container mx-auto px-6 py-2 flex justify-between items-center">
                    {/* Logo/Title */}
                    <div className="text-2xl font-bold flex items-center gap-2">
                        <a href="/" className="hover:text-green-600 transition">
                            <img src="/logo1.png" alt="logo" className="max-w-full h-auto"
                                style={{ maxWidth: '200px' }} />
                        </a>
                    </div>

                    {/* Desktop Navigation */}
                    <nav className="hidden md:flex items-center space-x-6">
                        <Link to="/about"
                        className="text-md hover:text-green-600 transition font-semibold">
                            About
                        </Link>
                        <Link to="/contact"
                        className="text-md hover:text-green-600 transition font-semibold">
                            Contact Us
                        </Link>

                        <Link to="/login">
                            <button className="font-semibold py-2 px-6 bg-slate-500 rounded-lg hover:bg-slate-700 hover:text-white transition-all duration-300">
                                Login
                            </button>
                        </Link>
                    </nav>

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden text-2xl focus:outline-none"
                        onClick={toggleMenu}
                        aria-expanded={menuOpen}
                    >
                        {menuOpen ? "\u2715" : "\u2630"} {/* Hamburger / Close Icon */}
                    </button>
                </div>

                {/* Mobile Navigation */}
                {menuOpen && (
                    <nav className="md:hidden bg-slate-100 rounded-xl shadow-lg mt-2">
                        <ul className="flex flex-col space-y-4 p-4">
                        <li>
                                <Link
                                    to="/about"
                                    className="block text-lg font-semibold hover:text-green-600 transition"
                                >
                                    About
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/contact"
                                    className="block text-lg font-semibold hover:text-green-600 transition"
                                >
                                    Contact Us
                                </Link>
                            </li>
                            <li>
                            <Link to="/login">
                            <button className="block text-lg font-semibold py-2 px-6 bg-slate-500 rounded-lg hover:bg-slate-700 hover:text-white transition-all duration-300">
                                Login
                            </button>
                        </Link>
                            </li>
                        </ul>
                    </nav>
                )}
            </header>
        </div>
    );
};

export default Navbar;
