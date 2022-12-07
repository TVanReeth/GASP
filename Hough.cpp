// File: Hough.cpp
// Author: Timothy Van Reeth <timothy.vanreeth@kuleuven.be>
// License: GPL-3+
// Description: Module that computes Hough functions
//              (based on a python script written by Vincent Prat)

#define _USE_MATH_DEFINES


#include <iostream>
#include <string>
#include <cmath>
#include <armadillo>

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html

// NOTE: the C++11 "auto" keyword is not recommended for use with Armadillo objects and functions


arma::vec eigenvalue_lambda(double nu, int l, int m, int npts = 400)
{
    // declaring variables
    int m_size, parity, pf, j_index, n_accepted_eigval, i_accepted_eigval;
    double cij, sij;
    arma::vec n, mu, s, denom, coeffs0, coeffs1, coeffs2, accepted_eigval;
    arma::mat full, d0, d1, d2, d0_other, d1_other, d0_inv, d1_dot, d2_dot, d0_inv_other, d1_dot_other, accepted_eigvec;
    arma::cx_vec eigval;
    arma::cx_mat eigvec;
    arma::uvec eigval_sorted_ind;

    // enforcing an even number of points
    m_size = (int)npts / 2;
    npts = m_size * 2;
    
    // define the parity
    parity = (l-m) % 2;
    
    // Calculate the interior/root points (mu_i = cos(((2i-1)Pi)/2N), where i = 1,...,N , N = total number of collocation points; see Wang et al. 2016)
    n = arma::linspace(0, m_size-1, m_size);
    mu = arma::cos((n + 0.5) * M_PI / npts);

    s = arma::sqrt(1 - arma::pow(mu,2));

    // define the coefficients of the differential equation for the radial Hough function (see Appendix A, Prat et al. 2019)
    denom = 1 - pow(nu,2) * arma::pow(mu,2);
    coeffs2 = arma::pow(s,2) / denom;
    coeffs1 = -2 * mu * (1 - pow(nu,2)) / arma::pow(denom,2);
    coeffs0 = nu * m * (1 + pow(nu,2) * arma::pow(mu,2)) / arma::pow(denom,2) - pow(m,2) / (arma::pow(s,2) % denom);

    // define the parity factor
    pf = m % 2;

    // Calculate the Chebyshev matrix (see Wang et al. 2016)
    full.zeros(m_size, m_size);
    d0.zeros(m_size, m_size);
    d1.zeros(m_size, m_size);
    d2.zeros(m_size, m_size);

    // d0 = chebyshev polynomial T_n (/times pf) ; d1 = first derivative with respect to mu ; d2 = second derivative with respect to mu; see also Boyd (2001)
    
    for (int i = 0; i < m_size; i++){
        for (int j = 0; j < m_size; j++){
            j_index = 2 * j + parity;     // cfr. 'n' in T_n

            cij = cos(M_PI*j_index/npts * (npts+i+0.5));
            sij = sin(M_PI*j_index/npts * (npts+i+0.5));

            if (pf == 1){
                d0(i,j) = cij * s(i);
                d1(i,j) = j_index * sij - cij * mu(i) / s(i);
                d2(i,j) = (-pow(j_index,2) * cij * pow(s(i),2) - j_index * sij * mu(i) * s(i) - cij) / pow(s(i),3);
            }
            else {
                d0(i,j) = cij;
                d1(i,j) = j_index * sij / s(i);
                d2(i,j) = j_index * (mu(i) * sij - j_index * cij * s(i)) / pow(s(i),3);
            }
        }
    }

    // multiply derivatives with inverse of T_n (d0) -----> obtain unity matrix coeffs0 accompanying term (see Full matrix projection below)
    d0_inv = arma::inv(d0);
    d1_dot = d1 * d0_inv;
    d2_dot = d2 * d0_inv;

    // Full matrix projection into Chebyshev space of differential equation for the radial Hough function (see Appendix A, Prat et al. 2019)
    full = (arma::diagmat(coeffs2) * d2_dot) + (arma::diagmat(coeffs1) * d1_dot) + arma::diagmat(coeffs0);
    
    // computation of eigenvalues/-functions --> for dense matrices!
    arma::eig_gen( eigval, eigvec, full );
    
    // remove complex eigenvalues and eigenvalues with positive real parts
    n_accepted_eigval = 0;

    for (int i = 0; i < m_size; i++){
        if ((imag(eigval(i)) == 0) || (real(eigval(i)) < 0) ){
            n_accepted_eigval++;
        }
    }

    accepted_eigval.zeros(n_accepted_eigval);
    accepted_eigvec.zeros(m_size,n_accepted_eigval);
    eigval_sorted_ind = arma::sort_index(eigval);
    i_accepted_eigval = 0;

    for (int i = 0; i < m_size; i++){
        if ((imag(eigval(eigval_sorted_ind(i))) == 0) || (real(eigval(eigval_sorted_ind(i))) < 0) ){
            i_accepted_eigval++;
            accepted_eigval(i_accepted_eigval) = real(eigval(eigval_sorted_ind(i)));
            accepted_eigvec(arma::span::all,i_accepted_eigval) = arma::real(eigvec(arma::span::all,eigval_sorted_ind(i)));
        }
    }
    
    return accepted_eigval;
}



arma::mat hough_function(double nu, int l, int m, int npts = 400, double est_lambda = 0)
{
    // declaring variables
    int m_size, parity, pf, j_index, n_accepted_eigval, i_accepted_eigval, ind_slct;
    double cij, sij, maxval, slct_eigval;
    arma::vec n, mu, s, denom, coeffs0, coeffs1, coeffs2, full_mu, full_s, accepted_eigval, hr, ht, hp, coeffs0_ht , coeffs1_ht, coeffs0_hp , coeffs1_hp; 
    arma::mat full, d0, d1, d2, d0_other, d1_other, d0_inv, d1_dot, d2_dot, d0_inv_other, d1_dot_other, accepted_eigvec, full_ht, full_hp, full_eigenfun;
    arma::cx_vec eigval;
    arma::cx_mat eigvec;
    arma::uvec eigval_sorted_ind;
    
    // enforcing an even number of points
    m_size = (int)npts / 2;
    npts = m_size * 2;
    
    // define the parity
    parity = (l-m) % 2;
    
    // Calculate the interior/root points (mu_i = cos(((2i-1)Pi)/2N), where i = 1,...,N , N = total number of collocation points; see Wang et al. 2016)
    n = arma::linspace(0, m_size-1, m_size);
    mu = arma::cos((n + 0.5) * M_PI / npts);

    s = arma::sqrt(1 - arma::pow(mu,2));
    
    // define the coefficients of the differential equation for the radial Hough function (see Appendix A, Prat et al. 2019)
    denom = 1 - pow(nu,2) * arma::pow(mu,2);
    coeffs2 = arma::pow(s,2) / denom;
    coeffs1 = -2 * mu * (1 - pow(nu,2)) / arma::pow(denom,2);
    coeffs0 = nu * m * (1 + pow(nu,2) * arma::pow(mu,2)) / arma::pow(denom,2) - pow(m,2) / (arma::pow(s,2) % denom);
    
    // define the parity factor
    pf = m % 2;
    
    // Calculate the Chebyshev matrix (see Wang et al. 2016)
    full.zeros(m_size, m_size);
    d0.zeros(m_size, m_size);
    d1.zeros(m_size, m_size);
    d2.zeros(m_size, m_size);

    // d0 = chebyshev polynomial T_n (/times pf) ; d1 = first derivative with respect to mu ; d2 = second derivative with respect to mu; see also Boyd (2001)
    
    for (int i = 0; i < m_size; i++){
        for (int j = 0; j < m_size; j++){
            j_index = 2 * j + parity;     // cfr. 'n' in T_n

            cij = cos(M_PI*j_index/npts * (npts+i+0.5));
            sij = sin(M_PI*j_index/npts * (npts+i+0.5));
            
            if (pf == 1){
                d0(i,j) = cij * s(i);
                d1(i,j) = j_index * sij - cij * mu(i) / s(i);
                d2(i,j) = (-pow(j_index,2) * cij * pow(s(i),2) - j_index * sij * mu(i) * s(i) - cij) / pow(s(i),3);
            }
            else {
                d0(i,j) = cij;
                d1(i,j) = j_index * sij / s(i);
                d2(i,j) = j_index * (mu(i) * sij - j_index * cij * s(i)) / pow(s(i),3);
            }
        }
    }
    
    // multiply derivatives with inverse of T_n (d0) -----> obtain unity matrix coeffs0 accompanying term (see Full matrix projection below)
    d0_inv = arma::inv(d0);
    d1_dot = d1 * d0_inv;
    d2_dot = d2 * d0_inv;
    
    // Full matrix projection into Chebyshev space of differential equation for the radial Hough function (see Appendix A, Prat et al. 2019)
    full = (arma::diagmat(coeffs2) * d2_dot) + (arma::diagmat(coeffs1) * d1_dot) + arma::diagmat(coeffs0);
    
    // computation of eigenvalues/-functions --> for dense matrices!
    arma::eig_gen( eigval, eigvec, full );

    // remove complex eigenvalues and eigenvalues with positive real parts
    n_accepted_eigval = 0;

    for (int i = 0; i < m_size; i++){
        if ((imag(eigval(i)) == 0) || (real(eigval(i)) < 0) ){
            n_accepted_eigval++;
        }
    }
    
    accepted_eigval.zeros(n_accepted_eigval);
    accepted_eigvec.zeros(m_size, n_accepted_eigval);
    eigval_sorted_ind = arma::sort_index(eigval);
    i_accepted_eigval = -1;
    
    for (int i = 0; i < m_size; i++){
        if ((imag(eigval(eigval_sorted_ind(i))) == 0) || (real(eigval(eigval_sorted_ind(i))) < 0) ){
            i_accepted_eigval++;
            accepted_eigval(i_accepted_eigval) = real(eigval(eigval_sorted_ind(i)));
            accepted_eigvec(arma::span::all, i_accepted_eigval) = arma::real(eigvec(arma::span::all,eigval_sorted_ind(i)));
        }
    }
    
    // if estimate for eigenvalue not provided, generate automatic estimate
    if (est_lambda == 0) {
        est_lambda = -l*(l+1); // eigenvalue in the non-rotating case provided as estimate
    }
    
    ind_slct = arma::index_min(arma::pow(accepted_eigval - est_lambda,2));
    slct_eigval = accepted_eigval(ind_slct);

    // eigenfunction for the radial Hough function differential equation = radial Hough function
    hr = arma::vectorise(accepted_eigvec(arma::span::all,ind_slct));

    // normalisation of the radial Hough function (hr)
    if (hr(m_size-1) < 0) { // if last point is negative, switch sign
        hr *= -1;
    }
    maxval = arma::max(arma::abs(hr));
    hr = hr / maxval;
    
    // compute ht (latitudinal Hough function) from hr: see Appendix A (Prat et al. 2019) where Hr' denotes a derivative with respect to theta,
    // which has to be converted to a derivative with respect to mu (=cos(theta)) in order to obtain coefficients below
    coeffs1_ht = - arma::pow(s,2) / denom;
    coeffs0_ht = - m * nu * mu / denom;
    full_ht = (arma::diagmat(coeffs1_ht) * d1_dot) + arma::diagmat(coeffs0_ht);  // projection into Chebyshev space
    ht = arma::vectorise((full_ht * hr) / s);
    
    // compute hp (azimuthal Hough function) from hr: see Appendix A (Prat et al. 2019), see comment above with respect to derivatives & coefficients
    coeffs1_hp = nu * ( mu % arma::pow(s,2) ) / denom;
    coeffs0_hp = m / denom;
    full_hp = (arma::diagmat(coeffs1_hp) * d1_dot) + arma::diagmat(coeffs0_hp); // projection into Chebyshev space
    hp = arma::vectorise((full_hp * hr) / s);
    
    // append the symmetric terms
    full_eigenfun.zeros(2*m_size,4);
    full_eigenfun(arma::span(0,m_size-1), 0) = mu;
    full_eigenfun(arma::span(m_size,2*m_size-1), 0) = arma::reverse(-mu);
    
    if (parity) {
        full_eigenfun(arma::span(0,m_size-1), 1) = hr;
        full_eigenfun(arma::span(m_size,2*m_size-1), 1) = arma::reverse(-hr);
        full_eigenfun(arma::span(0,m_size-1), 2) = ht;
        full_eigenfun(arma::span(m_size,2*m_size-1), 2) = arma::reverse(ht);
        full_eigenfun(arma::span(0,m_size-1), 3) = hp;
        full_eigenfun(arma::span(m_size,2*m_size-1), 3) = arma::reverse(-hp);
    } else {
        full_eigenfun(arma::span(0,m_size-1), 1) = hr;
        full_eigenfun(arma::span(m_size,2*m_size-1), 1) = arma::reverse(hr);
        full_eigenfun(arma::span(0,m_size-1), 2) = ht;
        full_eigenfun(arma::span(m_size,2*m_size-1), 2) = arma::reverse(-ht);
        full_eigenfun(arma::span(0,m_size-1), 3) = hp;
        full_eigenfun(arma::span(m_size,2*m_size-1), 3) = arma::reverse(hp);
    }
    
    return full_eigenfun;
}



arma::mat hough_functions_townsend(double nu, int k, int m, int npts = 400)
{
    // declaring variables
    int l, m_size, parity;
    arma::vec n, mu, s;
    arma::mat W;

    // enforcing an even number of points
    m_size = (int)npts / 2;
    npts = m_size * 2;
    
    // define the parity
    parity = (l-m) % 2;
    
    // Calculate the interior/root points (mu_i = cos(((2i-1)Pi)/2N), where i = 1,...,N , N = total number of collocation points; see Wang et al. 2016)
    n = arma::linspace(0, m_size-1, m_size);
    mu = arma::cos((n + 0.5) * M_PI / npts);

    s = arma::sqrt(1 - arma::pow(mu,2));

    W.zeros(m_size, m_size);

    for (j=1, j <= m_size, j++) {
        W(j-1, j-1) = (1 - m*nu/Lambda(l_(j, m, parity)) - pow(nu,2) * F(l_(j, m, parity), m, nu)) / Lambda(l_(j, m, parity));
    }

    for (j=1, j <= m_size-3, j++) {
        W(j-1,j) = -pow(nu,2) * J(m,l_(j, m, parity)+1) * J(m,l_(j, m, parity)+2) / (Lambda(l_(j, m, parity)+1) - m * nu);
        W(j,j-1) = W(j-1,j);
    }

    // I GOT TO HERE!
    // checking round and square brackets of array elements...
}


double l_(int j, int m, int parity) {
    if (parity == 0) {
        return abs(m) + 2*(j - 1);
    } else {
        return abs(m) + 2*(j - 1) + 1;
    }
}


double lk_(int j, int m, int parity) {
    if (parity == 1) {
        return abs(m) + 2*(j - 1);
    } else {
        return abs(m) + 2*(j - 1) + 1;
    }
}


double Lambda(double lj) {
    return lj*(lj + 1);
}


double J(int m, double lj) {
    if((lj <= abs(m)) || (lj == 1)) {
        return 0;
    } else {
        return sqrt((pow(lj,2) - pow(m,2))/(4*pow(lj,2) - 1));
    }
}


double F(double lj, int m, double nu){
    if (J(m, lj) == 0) {
        return (lj*(lj+2) * pow(J(m,lj+1),2)) / (pow((lj+1),2) * (1 - m*nu / Lambda(lj+1)));
    } else {
        return ((pow(lj,2) - 1) * pow(J(m,lj),2)) / (pow(lj,2)*(1-m*nu/Lambda(lj-1))) + (lj*(lj+2) * pow(J(m,lj+1),2)) / (pow(lj+1, 2) * (1 - m*nu / Lambda(lj+1)));
    }
}


// MAIN FUNCTION
int main(int argc, char **argv)
  {
        int k, l, m, npts;
        double nu, est_lambda, nu_mult;
        std::string main_dir, save_dir, filename, mode_ID, spin_str, num_str;
        arma::mat full_eigenfun;
        arma::field<std::string> hough_header(4);
        
        // basic input parameters
        main_dir = "/lhome/timothyv/FAMIAS/GASP/";
        npts = 1000;
        
        // reading in the command line arguments
        k = std::stoi(argv[1]);
        m = std::stoi(argv[2]);
        nu = std::stod(argv[3]);

        // hard-coded data structure
        save_dir = main_dir + "Data/Mode_geometries/";
        mode_ID = "_k" + std::to_string(k) + "m" + std::to_string(m);
        nu_mult = nu*10000;
        spin_str = "_nu" + std::to_string((int) nu_mult);
        num_str = "_npts" + std::to_string(npts);
        filename = save_dir + "gmode" + mode_ID + spin_str + num_str + ".csv";
        
        // calculating the Hough function
        l = abs(k) + abs(m);
        est_lambda = -l*(l+1);
        full_eigenfun = hough_function(nu, l, m, npts = npts, est_lambda = est_lambda);
        
        // Saving the mode geometry
        hough_header(0) = "mu";
        hough_header(1) = "Hr";
        hough_header(2) = "Ht";
        hough_header(3) = "Hp";
        full_eigenfun.save( arma::csv_name( filename, hough_header ) );
        
        return 0;
  }
