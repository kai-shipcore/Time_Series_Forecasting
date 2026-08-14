# Dashboard Requirements (Original Source)

Verbatim from the forecasting product and business owner, passed on by a coworker.
`dashboard/PAGES_BUILD_SPEC.md` was derived from this; when the two disagree, this is the
source of truth. Preserved here on 2026-07-24 because until now it only existed in chat
history.

One deliberate deviation: this document uses "Established Product" / "New Product" for the
model-group split. The codebase uses "short" / "long" instead (`docs/ML_FORECAST_DESIGN.md`'s
terminology), and every page in this dashboard follows that, not the wording below.

Second deviation: this document lists "Master SKU" as a column separate from the SKU
itself (see "2. Purchase Priority List" and "3. SKU Detail View" below). We checked the
real inventory tables (`ecommerce_data.coverland_inventory`, `coverland_mastersku` in the
Commerce/Supabase database, 2026-07-24) and confirmed their "master_sku" values are exact
string matches for our `unique_id` — 435 of 447 forecasted SKUs matched verbatim. Despite the
name, there is no coarser parent-product grouping anywhere in that system; "master SKU" and
"SKU" are the same identifier there. An earlier version of this dashboard computed a
locally-invented, coarser `master_sku` (stripping the trailing colour/size token off
`unique_id`) that did not correspond to anything real and has been removed. Every page shows
one SKU identifier, not two.

---

## Forecast Visualization and Inventory Optimization Dashboard

### Recommended Role

**Forecasting Dashboard Development & Validation Intern**

### Role Description

Develop a dashboard that transforms demand forecast outputs into clear, actionable
information for purchasing, inventory, and logistics teams.

The dashboard should combine forecasted demand with current inventory, preorder backlog,
confirmed inbound quantities, and recent sales performance so users can quickly identify
inventory risks and determine which SKUs require immediate action.

### Recommended Technology

Use Python Streamlit for the initial prototype. Streamlit is suitable because:

- The forecasting models are already developed in Python.
- Forecast outputs can be connected directly from CSV files or the database.
- Filters, tables, charts, and SKU-level detail pages can be built quickly.
- Model updates can be reflected in the dashboard with minimal additional work.
- It is appropriate for internal testing before moving the approved design into the
  production web application.

Recommended development sequence:

- Phase 1: Build an internal Streamlit prototype
- Phase 2: Conduct user testing with the forecasting and purchasing teams
- Phase 3: Transfer approved functionality into the production system

## Main Dashboard Requirements

### 1. Inventory Overview

Display key summary metrics such as:

- Total number of forecasted SKUs
- Number of preorder-priority SKUs
- Number of out-of-stock SKUs
- Number of best sellers at risk of stockout
- Total recommended order quantity
- Number of SKUs expected to stock out within 30 days

### 2. Purchase Priority List

Apply the company's purchasing priorities:

- Priority 1: Preorder
- Priority 2: No Stock
- Priority 3: Best Seller

The SKU list should include:

- Master SKU
- Product name
- Priority level
- Current available inventory
- Preorder backlog
- Recent 30-day sales
- Average daily sales
- Forecasted demand
- Confirmed inbound quantity
- Estimated stockout date
- Recommended order quantity
- Expected ETA
- Model group: Established Product or New Product

### 3. SKU Detail View

When a user selects a SKU, display:

- Historical actual sales
- Model forecast
- Spreadsheet forecast
- Current inventory
- Confirmed inbound quantities
- Preorder backlog
- Estimated stockout date
- Recommended order quantity calculation

Example:

```
Preorder Demand              120
Lead-Time Demand             350
Safety Stock                 150
Available Inventory         -100
Confirmed Inbound           -200
--------------------------------
Recommended Order Quantity   320
```

### 4. Forecast Accuracy View

Create a separate validation or administrator tab showing:

- Actual sales versus forecasted sales
- New model versus spreadsheet method
- Established-product model performance
- New-product model performance
- Seasonal test results
- Overforecasted and underforecasted SKUs

### 5. Data Quality Alerts

Include alerts for data issues that may reduce forecast reliability:

- Missing SKU mappings
- Unclear or incorrectly classified preorder transactions
- SKUs with zero inventory but an In Stock status
- SKUs without incoming containers
- New SKUs with no sales history
- Forecasted SKUs without an assigned size
- Status conflicts involving `-NEW-`, `-INV-`, or Discontinued items

## Responsibility Assignment

**Forecasting Dashboard Development & Validation Intern**

- Build the Streamlit dashboard.
- Visualize forecast results and historical sales.
- Implement SKU search, filtering, and detail views.
- Separate results for established and newer products.
- Display forecast accuracy and model comparison results.
- Identify and display data-quality exceptions.
- Validate dashboard values against the source forecast output and raw data.
- Document calculations, assumptions, and known limitations.

**Forecasting Product and Business Owner**

- Define which metrics and decisions the dashboard must support.
- Confirm the Preorder, No Stock, and Best Seller priority logic.
- Review the recommended order quantity calculation.
- Coordinate feedback from purchasing and forecasting users.
- Approve the functionality that should later move into the production application.

## Suggested Assignment

Develop a Streamlit-based inventory optimization dashboard that allows purchasing and
logistics teams to use the demand forecasting results in their daily operations. The
dashboard should display preorder, out-of-stock, and best-seller priorities together with
current inventory, confirmed inbound quantities, forecasted demand, estimated stockout
dates, and recommended purchase quantities. It should also provide SKU-level comparisons
between actual sales, the new forecasting model, and the existing spreadsheet method.
Include data-quality checks for preorder classification issues, missing SKU mappings,
insufficient sales history, and conflicting product statuses. Validate all displayed
results against the original model output and document the calculation logic and any
known limitations.

## Initial Version Completion Criteria

The first version should include:

- Import forecast results from CSV or the database.
- Display a priority-based SKU list.
- Provide search and filter functions.
- Show actual sales versus forecasted demand by SKU.
- Display the recommended order quantity calculation.
- Show important data-quality exceptions.
